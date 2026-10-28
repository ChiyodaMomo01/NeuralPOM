"""CPU contract/adjoint checks; CUDA execution is gated separately."""
import unittest

import torch

from neuralpom.cores.implicit_p0_fixed.strict_vertical import (
    _reference, strict_proft, strict_proft_compiled_backward, warmup_strict_proft_backward,
    is_strict_proft_backward_warm)


def fixture(dtype=torch.float64, kb=6, device='cpu'):
    generator = torch.Generator().manual_seed(817)
    def random(shape):
        return torch.rand(shape, generator=generator, dtype=dtype).to(device)
    shape = (3, 4)
    return [torch.linspace(0, -1, kb, dtype=dtype, device=device),
            10 + random((*shape, kb)), random(shape)*1e-5, random(shape)+10,
            100 + random(shape)*20, random(shape)*.1,
            torch.full((kb,), 1/(kb-1), dtype=dtype, device=device),
            torch.full((kb,), 1/(kb-1), dtype=dtype, device=device),
            random(shape)*1e-4, random((*shape, kb))*1e-3]


def call(inputs, nbc=1, scratch=None, ntp=2, function=strict_proft):
    z,f,wf,fs,h,etf,dz,dzz,swrad,kh = inputs
    im,jm,kb = f.shape
    return function(z,f,wf,fs,nbc,scratch,scratch,scratch,scratch,scratch,
                        h,etf,1200.,dz,dzz,swrad,ntp,im,jm,kb,kb-1,kb-2,kh,1e-6)


class StrictVerticalTest(unittest.TestCase):
    def test_optional_compiled_adjoint_requires_external_warmup(self):
        from functools import partial
        from torch.utils.checkpoint import checkpoint
        inputs = [value.requires_grad_() for value in fixture(kb=5)]
        self.assertFalse(call(inputs, function=is_strict_proft_backward_warm))
        with self.assertRaisesRegex(RuntimeError, 'outside gradient checkpoint'):
            call(inputs, function=strict_proft_compiled_backward)
        call(inputs, function=partial(warmup_strict_proft_backward, backend='aot_eager'))
        self.assertTrue(call(inputs, function=is_strict_proft_backward_warm))
        self.assertFalse(call(inputs, nbc=2, function=is_strict_proft_backward_warm))
        for full_gradients in (True, False):
            copied = [value.detach().clone().requires_grad_(full_gradients or i == 1)
                      for i, value in enumerate(inputs)]
            eager = [value.detach().clone().requires_grad_(value.requires_grad) for value in copied]
            def objective(values, function):
                outputs = call(values, function=function)
                return sum((outputs[i] * (i+1)).sum() for i in (0,4,5,6,7,8))
            loss = checkpoint(lambda *values: objective(values, strict_proft_compiled_backward),
                              *copied, use_reentrant=False)
            reference = objective(eager, strict_proft)
            active = [value for value in copied if value.requires_grad]
            actual = torch.autograd.grad(loss, active, allow_unused=True)
            expected = torch.autograd.grad(reference, [value for value in eager if value.requires_grad],
                                           allow_unused=True)
            torch.testing.assert_close(loss, reference, rtol=0, atol=0)
            for a,b in zip(actual,expected):
                self.assertEqual(a is None,b is None)
                if a is not None:
                    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        self.assertTrue(all(value.grad is None for value in inputs))

    def test_all_boundaries_preserve_full_return_and_inputs(self):
        for nbc in (1,2,3,4):
            with self.subTest(nbc=nbc):
                inputs=fixture(dtype=torch.float32)
                original=[x.clone() for x in inputs]
                scratch=torch.full_like(inputs[1], float('nan'), requires_grad=True)
                actual=call(inputs,nbc,scratch)
                reference=_reference(inputs,(nbc,1200.,2,1e-6))
                self.assertEqual(len(actual),9)
                self.assertIs(actual[1],inputs[2])
                self.assertIs(actual[2],inputs[3])
                self.assertEqual(actual[3],nbc)
                for x,y in zip([actual[i] for i in (0,4,5,6,7,8)],reference):
                    self.assertTrue(torch.equal(x,y))
                self.assertTrue(torch.equal(actual[0][...,-1],inputs[1][...,-1]))
                for x,y in zip(inputs,original):
                    self.assertTrue(torch.equal(x,y))
                self.assertTrue(torch.isnan(scratch).all())

    def test_all_input_and_scratch_output_gradients(self):
        for nbc in (1,2,3,4):
            with self.subTest(nbc=nbc):
                inputs=[x.requires_grad_() for x in fixture()]
                copied=[x.detach().clone().requires_grad_() for x in inputs]
                actual=call(inputs,nbc)
                reference=_reference(copied,(nbc,1200.,2,1e-6))
                outputs=[actual[i] for i in (0,4,5,6,7,8)]
                weights=[torch.sin(torch.arange(x.numel(),dtype=x.dtype).reshape(x.shape)+i) for i,x in enumerate(outputs)]
                objective=sum((x*w).sum() for x,w in zip(outputs,weights))+actual[1].sum()+2*actual[2].sum()
                expected=sum((x*w).sum() for x,w in zip(reference,weights))+copied[2].sum()+2*copied[3].sum()
                gradients=torch.autograd.grad(objective,inputs,allow_unused=True)
                expected_gradients=torch.autograd.grad(expected,copied,allow_unused=True)
                for got,want in zip(gradients,expected_gradients):
                    self.assertEqual(got is None,want is None)
                    if got is not None:
                        self.assertTrue(torch.isfinite(got).all())
                        # Identity returns accumulate outside the custom node;
                        # their sum can differ by one float64 rounding unit.
                        torch.testing.assert_close(got,want,rtol=1e-14,atol=1e-14)

    def test_noncontiguous_inputs_and_finite_difference(self):
        inputs=[x.requires_grad_() for x in fixture(kb=4)]
        # Shape remains unchanged; the wrapper must preserve the transpose path.
        inputs[1]=inputs[1].transpose(0,1).contiguous().transpose(0,1)
        def objective(field,depth,diffusivity):
            current=list(inputs)
            current[1],current[4],current[9]=field,depth,diffusivity
            return call(current)[0]
        self.assertTrue(torch.autograd.gradcheck(objective,(inputs[1],inputs[4],inputs[9]),
                                               eps=1e-5,atol=2e-6,rtol=2e-5,fast_mode=True))

    def test_surface_and_uniform_column_solution(self):
        inputs=fixture()
        inputs[1]=torch.full_like(inputs[1],7.)
        inputs[2]=torch.zeros_like(inputs[2])
        output=call(inputs,1)[0]
        torch.testing.assert_close(output,inputs[1],rtol=1e-15,atol=1e-15)
        for nbc in (3,4):
            output=call(inputs,nbc)[0]
            torch.testing.assert_close(output[...,0],inputs[3],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
