import { defineConfig } from 'vitepress'

const sections = [
  ['Guide', [
    ['Quick start', '/guide/quickstart'], ['Installation', '/guide/installation'],
    ['Configuration', '/guide/configuration'], ['Reproduction', '/reproduction']
  ]],
  ['Dynamical cores', [
    ['Overview', '/cores/overview'], ['Explicit', '/cores/explicit'],
    ['Semi-implicit', '/cores/implicit'], ['Differentiation and restart', '/cores/gradient-safety']
  ]],
  ['Physics', [
    ['Governing equations', '/physics/equations'],
    ['Mode splitting', '/physics/barotropic-baroclinic'], ['Turbulence', '/physics/turbulence']
  ]],
  ['Numerics', [
    ['Arakawa C grid', '/numerics/arakawa-c-grid'],
    ['Time stepping', '/numerics/time-stepping'], ['Elliptic solver', '/numerics/elliptic-solver']
  ]],
  ['Hybrid model', [
    ['Overview', '/hybrid/overview'], ['CorrectorNet', '/hybrid/corrector-net'],
    ['Staged training', '/hybrid/training'], ['Supervision', '/hybrid/supervision'],
    ['Posttrain interfaces', '/v1/posttrain'], ['Daily AI baselines', '/ai_daily_baselines']
  ]],
  ['Data and cases', [
    ['Data pipeline', '/data/pipeline'], ['Reference format', '/data/format'],
    ['Cases', '/cases/overview'], ['Double gyre', '/cases/double-gyre']
  ]],
  ['Development', [
    ['Architecture', '/developer/architecture'], ['Contributing', '/developer/contributing'],
    ['API and entry points', '/appendix/api'], ['References', '/appendix/references']
  ]]
]

export default defineConfig({
  title: 'NeuralPOM',
  description: 'Differentiable ocean dynamics and staged learning',
  head: [['link', { rel: 'icon', href: '/logo.png' }]],
  markdown: { math: true, lineNumbers: true },
  themeConfig: {
    logo: '/logo.png',
    search: { provider: 'local' },
    nav: [
      { text: 'Guide', link: '/guide/quickstart' },
      { text: 'Reproduction', link: '/reproduction' },
      { text: 'Physics', link: '/physics/equations' }
    ],
    sidebar: sections.map(([text, items]) => ({
      text, collapsed: false, items: items.map(([text, link]) => ({ text, link }))
    })),
    socialLinks: [{ icon: 'github', link: 'https://github.com/ChiyodaMomo01/NeuralPOM' }],
    footer: { copyright: 'NeuralPOM Contributors' }
  }
})
