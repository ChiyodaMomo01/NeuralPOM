import { defineConfig } from "vitepress";

const sections = [
  [
    "Start here",
    [
      ["Reading roadmap", "/guide/roadmap"],
      ["Read the paper", "/paper"],
      ["Quick start", "/guide/quickstart"],
      ["Installation", "/guide/installation"],
      ["Configuration", "/guide/configuration"],
      ["Reproduction", "/reproduction"],
    ],
  ],
  [
    "Dynamical cores",
    [
      ["Overview", "/cores/overview"],
      ["Explicit", "/cores/explicit"],
      ["Semi-implicit", "/cores/implicit"],
      ["Differentiation and restart", "/cores/gradient-safety"],
    ],
  ],
  [
    "Physics",
    [
      ["Governing equations", "/physics/equations"],
      ["Mode splitting", "/physics/barotropic-baroclinic"],
      ["Turbulence", "/physics/turbulence"],
    ],
  ],
  [
    "Numerics",
    [
      ["Arakawa C grid", "/numerics/arakawa-c-grid"],
      ["Time stepping", "/numerics/time-stepping"],
      ["Elliptic solver", "/numerics/elliptic-solver"],
    ],
  ],
  [
    "Hybrid model",
    [
      ["Overview", "/hybrid/overview"],
      ["CorrectorNet", "/hybrid/corrector-net"],
      ["Staged training", "/hybrid/training"],
      ["Supervision", "/hybrid/supervision"],
      ["Density-normal damping", "/hybrid/density-normal-damping"],
      ["Posttrain interfaces", "/v1/posttrain"],
      ["Daily AI baselines", "/ai_daily_baselines"],
    ],
  ],
  [
    "Data and cases",
    [
      ["Data pipeline", "/data/pipeline"],
      ["Reference format", "/data/format"],
      ["Cases", "/cases/overview"],
      ["Double gyre", "/cases/double-gyre"],
    ],
  ],
  [
    "Development",
    [
      ["Architecture", "/developer/architecture"],
      ["Contributing", "/developer/contributing"],
      ["API and entry points", "/appendix/api"],
      ["References", "/appendix/references"],
    ],
  ],
];

export default defineConfig({
  title: "NeuralPOM",
  description:
    "A research field guide to differentiable ocean dynamics, constrained neural corrections, and staged learning.",
  cleanUrls: true,
  sitemap: { hostname: "https://neural-pom.vercel.app" },
  head: [
    ["link", { rel: "icon", href: "/logo.png" }],
    ["meta", { name: "theme-color", content: "#fcfaf7" }],
    ["meta", { property: "og:type", content: "website" }],
    [
      "meta",
      { property: "og:title", content: "NeuralPOM — A research field guide" },
    ],
    [
      "meta",
      {
        property: "og:description",
        content:
          "Ocean dynamics. Learned corrections. Explore the science, implementation, and reproduction workflow.",
      },
    ],
    [
      "meta",
      {
        property: "og:image",
        content:
          "https://neural-pom.vercel.app/figures/developed_surface_fields.png",
      },
    ],
    [
      "link",
      {
        rel: "preload",
        href: "/fonts/serif.woff2",
        as: "font",
        type: "font/woff2",
        crossorigin: "",
      },
    ],
    [
      "link",
      {
        rel: "preload",
        href: "/fonts/sans.woff2",
        as: "font",
        type: "font/woff2",
        crossorigin: "",
      },
    ],
  ],
  markdown: { math: true, lineNumbers: true },
  themeConfig: {
    logo: "/logo.png",
    search: { provider: "local" },
    nav: [
      { text: "The guide", link: "/guide/roadmap", activeMatch: "/guide/" },
      { text: "The method", link: "/hybrid/overview", activeMatch: "/hybrid/" },
      { text: "Reproduction", link: "/reproduction" },
      { text: "Paper", link: "/paper" },
    ],
    sidebar: sections.map(([text, items]) => ({
      text,
      collapsed: !["Start here", "Hybrid model"].includes(text),
      items: items.map(([text, link]) => ({ text, link })),
    })),
    socialLinks: [
      { icon: "github", link: "https://github.com/ChiyodaMomo01/NeuralPOM" },
    ],
    outline: { level: [2, 3], label: "On this page" },
    editLink: {
      pattern:
        "https://github.com/ChiyodaMomo01/NeuralPOM/edit/main/docs/:path",
      text: "Improve this page on GitHub",
    },
    footer: {
      message:
        "Differentiable ocean dynamics · Constrained neural corrections · Staged learning",
      copyright: "NeuralPOM · Code licensed under Apache-2.0",
    },
  },
});
