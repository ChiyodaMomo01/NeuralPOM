<script setup>
const sections = [
  {
    title: "Understand the model",
    items: [
      {
        title: "Physical framework",
        link: "/physics/equations",
        description: "Primitive equations, grid, and dynamical cores.",
      },
      {
        title: "Constrained corrections",
        link: "/hybrid/overview",
        description: "How neural increments update the ocean state.",
      },
      {
        title: "Staged learning",
        link: "/hybrid/training",
        description: "Pretrain, Midtrain, and Posttrain.",
      },
      {
        title: "Density-normal damping",
        link: "/hybrid/density-normal-damping",
        description: "The tracer projection, explained interactively.",
      },
    ],
  },
  {
    title: "Work with the code",
    items: [
      {
        title: "Installation and quick start",
        link: "/guide/quickstart",
        description: "Set up a configuration and inspect the training plan.",
      },
      {
        title: "Data and calibration",
        link: "/data/pipeline",
        description: "Reference ensembles, preparation, and model inputs.",
      },
      {
        title: "Reproduce the experiments",
        link: "/reproduction",
        description: "Training, frozen forecasts, and source tables.",
      },
      {
        title: "Implementation guide",
        link: "/developer/architecture",
        description: "Find the relevant modules and entry points.",
      },
    ],
  },
];
</script>

<template>
  <main class="guide-home">
    <header class="guide-intro">
      <h1>NeuralPOM</h1>
      <p class="guide-subtitle">
        Differentiable ocean dynamics and staged learning.
      </p>
      <p class="guide-description">
        NeuralPOM combines the Princeton Ocean Model with constrained neural
        corrections. This guide connects the physical formulation, learning
        method, and code to the experiments in the paper.
      </p>
      <div class="guide-actions">
        <a class="guide-start" href="/guide/quickstart">Get started</a>
        <a href="/guide/roadmap">Reading roadmap</a>
        <a href="/paper">Read the paper</a>
      </div>
    </header>
    <div class="guide-sections">
      <section
        v-for="section in sections"
        :key="section.title"
        :aria-label="section.title"
      >
        <h2>{{ section.title }}</h2>
        <ul>
          <li v-for="item in section.items" :key="item.link">
            <a :href="item.link">{{ item.title }}</a>
            <p>{{ item.description }}</p>
          </li>
        </ul>
      </section>
    </div>
    <footer class="guide-resources">
      <p>Research resources</p>
      <nav aria-label="Research resources">
        <a href="https://github.com/ChiyodaMomo01/NeuralPOM">GitHub</a>
        <a href="https://www.modelscope.cn/datasets/chiyodamomo01/NeuralPOM"
          >Dataset</a
        >
        <a
          href="https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/paper/NeuralPOM.pdf"
          >Manuscript PDF</a
        >
        <a
          href="https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/paper/data"
          >Source tables</a
        >
      </nav>
    </footer>
  </main>
</template>
