import math
import torch


torch.set_default_dtype(torch.float32)


class BaseCore:
    def __init__(self, im, jm):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # =================================================================
        # 0. User Configuration: Select Problem ID
        # =================================================================
        # NeuralPOM is intentionally focused on the Double-gyre mainline.
        self.iproblem = getattr(self, "_requested_problem_id", 4) 
        if self.iproblem != 4:
            raise ValueError(
                f"Only the Double-gyre problem_id=4 is supported; got {self.iproblem}."
            )
        self.im = im
        self.jm = jm

        # =================================================================
        # 1. Grid Dimensions
        # =================================================================
        self.kb = 16
        
        # Derived indices
        self.imm1 = self.im - 1
        self.imm2 = self.im - 2
        self.jmm1 = self.jm - 1
        self.jmm2 = self.jm - 2
        self.kbm1 = self.kb - 1
        self.kbm2 = self.kb - 2

        # Scalars & Constants (Initialize to 0)
        self.alpha  = 0.0; self.dte   = 0.0; self.dti   = 0.0; self.dti2  = 0.0
        self.grav   = 0.0; self.hmax  = 0.0; self.kappa = 0.0; self.pi    = 0.0
        self.ramp   = 0.0; self.rfe   = 0.0; self.rfn   = 0.0; self.rfs   = 0.0
        self.rfw    = 0.0; self.rhoref= 0.0; self.sbias = 0.0; self.slmax = 0.0
        self.small  = 0.0; self.tbias = 0.0; self.time  = 0.0; self.tprni = 0.0
        self.umol   = 0.0; self.vmaxl = 0.0

        self.iint = 0; self.iprint = 0; self.iskp = 0; self.jskp = 0
        self.kl1 = 0;  self.kl2 = 0;    self.mode = 0; self.ntp = 0

        self.source = 'pom2k Double Gyre'
        self.netcdf_file = 'double_gyre_R4.nc'
        self.title = 'Wind-Driven Double Gyre'

        # =================================================================
        # 2. Physics & Time Configuration
        # =================================================================
        self.small = 1.e-9
        self.pi = math.atan(1.0) * 4.0
        
        self.mode = 3           # 3D Calculation
        self.nadv = 1           # Advection scheme (1: Centered, 2: Smolarkiewicz)
        self.nitera = 2
        self.sw = 0.50
        self.nread = 0
        self.lramp = True       # Enable ramp for smooth start
        
        self.rhoref = 1035.0
        self.hmax = 2000.0
        self.dte = 30.0
        self.isplit = 30
        self.days = 3650.0
        self.prtd1 = 1.0
        self.grav = 9.806
        self.kappa = 0.40
        self.z0b = 0.010
        self.cbcmin = 0.00250
        self.cbcmax = 1.0
        self.horcon = 0.20
        self.tprni = 0.20
        self.umol = 2.e-5
        self.vmaxl = 100.0
        self.slmax = 2.0
        self.kl1 = 2
        self.kl2 = 2

        # Common Derived Time Parameters
        self.swtch = 99999.0     
        self.iskp = 4
        self.jskp = 3
        
        self.ntp = 2
        self.nbct = 1
        self.nbcs = 1
        self.ispadv = 5
        # Stronger Asselin filter at large dt: damp leapfrog computational mode to avoid 2dx chessboard.
        self.smoth = 0.150
        # The implicit external mode is unconditionally stable for alpha >= 0.25.
        # The explicit core used alpha=0.225 under a CFL restriction; the
        # implicit core keeps alpha in a more stable range while staying close
        # to the original phase behavior.
        self.alpha = 0.333
        self.aam_init = 500.0
        self.aam_min = 0.0
        self.ramp = 0.0
        
        self.tatm = 0.0
        self.satm = 0.0
        
        self.io = torch.zeros(100, dtype=torch.int32)
        self.jo = torch.zeros(100, dtype=torch.int32)
        self.ko = torch.zeros(100, dtype=torch.int32)

        self.dti = self.dte * self.isplit
        self.dte2 = self.dte * 2.0
        self.dti2 = self.dti * 2.0
        
        self.iend = max(math.floor(self.days * 24.0 * 3600.0 / self.dti + 0.5), 2)
        self.iprint = math.floor(self.prtd1 * 24.0 * 3600.0 / self.dti + 0.5)
        self.iswtch = math.floor(self.swtch * 24.0 * 3600.0 / self.dti + 0.5)
        
        self.ispi = 1.0 / self.isplit
        self.isp2i = 1.0 / (2.0 * self.isplit)

        # ====================
        # Array Initialization (Standard)
        # ====================
        # Keep these tensors non-leaf so later in-place updates remain valid.
        def mk_zeros(shape):
            return torch.zeros(shape, device=self.device, requires_grad=False)

        # 1D Arrays
        self.dz  = mk_zeros((self.kb,))
        self.dzz = mk_zeros((self.kb,))
        self.z   = mk_zeros((self.kb,))
        self.zz  = mk_zeros((self.kb,))

        # 2D Arrays
        shape_2d = (self.im, self.jm)
        self.aam2d = mk_zeros(shape_2d); self.advua = mk_zeros(shape_2d); self.advva = mk_zeros(shape_2d)
        self.adx2d = mk_zeros(shape_2d); self.ady2d = mk_zeros(shape_2d); self.art   = mk_zeros(shape_2d)
        self.aru   = mk_zeros(shape_2d); self.arv   = mk_zeros(shape_2d); self.cbc   = mk_zeros(shape_2d)
        self.cor   = mk_zeros(shape_2d); self.d     = mk_zeros(shape_2d); self.drx2d = mk_zeros(shape_2d)
        self.dry2d = mk_zeros(shape_2d); self.dt    = mk_zeros(shape_2d); self.dum   = mk_zeros(shape_2d)
        self.dvm   = mk_zeros(shape_2d); 
        
        # FIX: requires_grad=False
        self.dx    = torch.ones(shape_2d, device=self.device, requires_grad=False)
        self.dy    = torch.ones(shape_2d, device=self.device, requires_grad=False)
        self.h     = torch.ones(shape_2d, device=self.device, requires_grad=False)

        self.east_c = mk_zeros(shape_2d); self.east_e = mk_zeros(shape_2d)
        self.east_u = mk_zeros(shape_2d); self.east_v = mk_zeros(shape_2d); self.e_atmos = mk_zeros(shape_2d)
        self.egb    = mk_zeros(shape_2d); self.egf    = mk_zeros(shape_2d); self.el      = mk_zeros(shape_2d)
        self.elb    = mk_zeros(shape_2d); self.elf    = mk_zeros(shape_2d); self.et      = mk_zeros(shape_2d)
        self.etb    = mk_zeros(shape_2d); self.etf    = mk_zeros(shape_2d); self.fluxua  = mk_zeros(shape_2d)
        self.fluxva = mk_zeros(shape_2d); self.fsm    = mk_zeros(shape_2d) 
        
        self.north_c = mk_zeros(shape_2d); self.north_e = mk_zeros(shape_2d)
        self.north_u = mk_zeros(shape_2d); self.north_v = mk_zeros(shape_2d); self.psi = mk_zeros(shape_2d)
        self.rot     = mk_zeros(shape_2d); self.ssurf   = mk_zeros(shape_2d); self.swrad = mk_zeros(shape_2d)
        self.vfluxb  = mk_zeros(shape_2d); self.tps     = mk_zeros(shape_2d); self.tsurf = mk_zeros(shape_2d)
        self.ua      = mk_zeros(shape_2d); self.vfluxf  = mk_zeros(shape_2d); self.uab   = mk_zeros(shape_2d)
        self.uaf     = mk_zeros(shape_2d); self.utb     = mk_zeros(shape_2d); self.utf   = mk_zeros(shape_2d)
        self.va      = mk_zeros(shape_2d); self.vab     = mk_zeros(shape_2d); self.vaf   = mk_zeros(shape_2d)
        self.vtb     = mk_zeros(shape_2d); self.vtf     = mk_zeros(shape_2d); self.wssurf= mk_zeros(shape_2d)
        self.wtsurf  = mk_zeros(shape_2d); self.wubot   = mk_zeros(shape_2d); self.wusurf= mk_zeros(shape_2d)
        self.wvbot   = mk_zeros(shape_2d); self.wvsurf  = mk_zeros(shape_2d)

        # 3D Arrays
        shape_3d = (self.im, self.jm, self.kb)
        self.aam   = mk_zeros(shape_3d); self.advx  = mk_zeros(shape_3d); self.advy  = mk_zeros(shape_3d)
        self.a     = mk_zeros(shape_3d); self.c     = mk_zeros(shape_3d); self.drhox = mk_zeros(shape_3d)
        self.drhoy = mk_zeros(shape_3d); self.dtef  = mk_zeros(shape_3d); self.ee    = mk_zeros(shape_3d)
        self.gg    = mk_zeros(shape_3d); self.kh    = mk_zeros(shape_3d); self.km    = mk_zeros(shape_3d)
        self.kq    = mk_zeros(shape_3d); self.l     = mk_zeros(shape_3d); self.q2b   = mk_zeros(shape_3d)
        self.q2    = mk_zeros(shape_3d); self.q2lb  = mk_zeros(shape_3d); self.q2l   = mk_zeros(shape_3d)
        self.rho   = mk_zeros(shape_3d); self.rmean = mk_zeros(shape_3d); self.sb    = mk_zeros(shape_3d)
        self.sclim = mk_zeros(shape_3d); self.s     = mk_zeros(shape_3d); self.tb    = mk_zeros(shape_3d)
        self.tclim = mk_zeros(shape_3d); self.t     = mk_zeros(shape_3d); self.ub    = mk_zeros(shape_3d)
        self.uf    = mk_zeros(shape_3d); self.u     = mk_zeros(shape_3d); self.vb    = mk_zeros(shape_3d)
        self.vf    = mk_zeros(shape_3d); self.v     = mk_zeros(shape_3d); self.w     = mk_zeros(shape_3d)
        self.zflux = mk_zeros(shape_3d)

        # Boundary Arrays
        self.ele = mk_zeros((self.jm,)); self.eln = mk_zeros((self.im,))
        self.els = mk_zeros((self.im,)); self.elw = mk_zeros((self.jm,))
        
        self.sbe = mk_zeros((self.jm, self.kb)); self.sbn = mk_zeros((self.im, self.kb))
        self.sbs = mk_zeros((self.im, self.kb)); self.sbw = mk_zeros((self.jm, self.kb))
        self.tbe = mk_zeros((self.jm, self.kb)); self.tbn = mk_zeros((self.im, self.kb))
        self.tbs = mk_zeros((self.im, self.kb)); self.tbw = mk_zeros((self.jm, self.kb))
        
        self.uabe= mk_zeros((self.jm,)); self.uabw = mk_zeros((self.jm,))
        self.ube = mk_zeros((self.jm, self.kb)); self.ubw = mk_zeros((self.jm, self.kb))
        self.vabn= mk_zeros((self.im,)); self.vabs = mk_zeros((self.im,))
        self.vbn = mk_zeros((self.im, self.kb)); self.vbs = mk_zeros((self.im, self.kb))
        self.curv = torch.zeros((self.im, self.jm, self.kb), device=self.device)
