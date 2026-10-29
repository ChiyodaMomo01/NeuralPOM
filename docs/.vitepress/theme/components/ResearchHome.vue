<script setup>
const chapters = [
  {
    n: "01",
    title: "The physical foundation",
    desc: "Primitive equations, a staggered grid, and two ways to advance the free surface.",
    link: "/physics/equations",
    meta: "PHYSICS & NUMERICS",
  },
  {
    n: "02",
    title: "Learning within the model",
    desc: "Follow a neural correction from network output to a consistent ocean state.",
    link: "/hybrid/overview",
    meta: "HYBRID METHOD",
  },
  {
    n: "03",
    title: "From endpoints to trajectories",
    desc: "Understand what changes across Pretrain, Midtrain, and Posttrain.",
    link: "/hybrid/training",
    meta: "STAGED LEARNING",
  },
  {
    n: "04",
    title: "From paper to experiment",
    desc: "Match configurations, reference data, frozen weights, and forecast diagnostics.",
    link: "/reproduction",
    meta: "REPRODUCTION",
  },
];
</script>

<template>
  <div class="research-home">
    <section class="research-hero" aria-labelledby="hero-title">
      <div class="hero-copy">
        <p class="eyebrow">
          <span class="small-rule"></span> THE NEURALPOM FIELD GUIDE
        </p>
        <h1 id="hero-title">
          Ocean dynamics.<br /><em>Learned corrections.</em>
        </h1>
        <p class="hero-deck">
          A familiar physical framework.<br />A new way to learn within it.
        </p>
        <p class="hero-description">
          Explore a hybrid ocean forecast model that brings differentiable
          Princeton Ocean Model dynamics together with constrained neural
          corrections and staged learning.
        </p>
        <div class="hero-actions">
          <a class="primary-link" href="/guide/roadmap"
            >Explore the guide <span aria-hidden="true">↗</span></a
          >
          <a class="text-link" href="/paper"
            >Read the research <span aria-hidden="true">→</span></a
          >
        </div>
        <p class="hero-note">
          OPEN SOURCE <span>·</span> PYTORCH <span>·</span> APACHE 2.0
        </p>
      </div>
      <figure class="ocean-plate">
        <div class="plate-top">
          <span>01 / THE DEVELOPED OCEAN</span><span>DOUBLE GYRE</span>
        </div>
        <div
          class="ocean-window"
          role="img"
          aria-label="Surface temperature panel from the NeuralPOM manuscript, showing the developed double-gyre circulation"
        >
          <img
            src="/figures/developed_surface_fields.png"
            alt=""
            fetchpriority="high"
            width="2925"
            height="2700"
          />
          <div class="ocean-coordinate top">REFERENCE FIELD</div>
          <div class="ocean-coordinate bottom">TEMPERATURE</div>
        </div>
        <figcaption>
          <span>Surface temperature · developed reference state</span
          ><a
            href="/paper#the-reference-ocean"
            aria-label="View the full reference figure and its source"
            >View full figure ↗</a
          >
        </figcaption>
      </figure>
    </section>

    <div class="research-strip" aria-label="Model overview">
      <div>
        <span class="strip-value">Two</span
        ><span>physical cores<br />Explicit & semi-implicit</span>
      </div>
      <div>
        <span class="strip-value">Three</span
        ><span>learning stages<br />endpoints → trajectories</span>
      </div>
      <div>
        <span class="strip-value">One</span
        ><span>evolving ocean state<br />physics → correction → physics</span>
      </div>
    </div>

    <section class="chapter-section" aria-labelledby="chapters-title">
      <div class="section-heading">
        <div>
          <p class="eyebrow">A PATH THROUGH THE METHOD</p>
          <h2 id="chapters-title">Understand. Run. Investigate.</h2>
        </div>
        <a class="text-link" href="/guide/quickstart"
          >Go straight to the code <span aria-hidden="true">↗</span></a
        >
      </div>
      <div class="chapter-list">
        <a
          v-for="chapter in chapters"
          :key="chapter.n"
          :href="chapter.link"
          class="chapter-row"
        >
          <span class="chapter-number">{{ chapter.n }}</span>
          <div class="chapter-title">
            <span class="eyebrow">{{ chapter.meta }}</span>
            <h3>{{ chapter.title }}</h3>
          </div>
          <p>{{ chapter.desc }}</p>
          <span class="chapter-arrow" aria-hidden="true">↗</span>
        </a>
      </div>
    </section>

    <section class="method-feature" aria-labelledby="method-title">
      <div>
        <p class="eyebrow">THE CONNECTION THAT MATTERS</p>
        <h2 id="method-title">
          A correction changes<br /><em>what happens next.</em>
        </h2>
        <p>
          The network updates velocity, temperature, and salinity at physical
          endpoints. The ocean model carries those changes forward. Training
          therefore has to account for both the correction and the dynamics it
          sets in motion.
        </p>
        <a class="text-link" href="/hybrid/training"
          >Follow the gradient paths <span aria-hidden="true">→</span></a
        >
      </div>
      <div
        class="flow-figure"
        role="img"
        aria-label="A complete ocean state advances through physical dynamics, receives a constrained correction, and continues into the next physical interval"
      >
        <div class="flow-step">
          <span>01</span>
          <div>
            <strong>Advance the ocean</strong
            ><small>Differentiable POM dynamics</small>
          </div>
          <span aria-hidden="true">↓</span>
        </div>
        <div class="flow-step active">
          <span>02</span>
          <div>
            <strong>Correct the endpoint</strong
            ><small>Learned increments · physical constraints</small>
          </div>
          <span aria-hidden="true">↓</span>
        </div>
        <div class="flow-step">
          <span>03</span>
          <div>
            <strong>Continue the trajectory</strong
            ><small>Updated fields, histories, and density</small>
          </div>
          <span aria-hidden="true">↻</span>
        </div>
        <p>One hour or one day between corrections</p>
      </div>
    </section>

    <section class="resource-section" aria-labelledby="resources-title">
      <div class="section-heading">
        <div>
          <p class="eyebrow">KEEP THE EVIDENCE CLOSE</p>
          <h2 id="resources-title">Paper, code, and data.</h2>
        </div>
        <p>Read the method.<br />Trace it to the implementation.</p>
      </div>
      <div class="resource-grid">
        <a href="/paper"
          ><span class="eyebrow">01 / MANUSCRIPT</span>
          <h3>New wine in <br />an old bottle.</h3>
          <p>The scientific argument, figures, and source tables.</p>
          <span class="text-link">Explore the paper ↗</span></a
        >
        <a href="https://github.com/ChiyodaMomo01/NeuralPOM"
          ><span class="eyebrow">02 / IMPLEMENTATION</span>
          <h3>Open the <br />model.</h3>
          <p>
            Physical solvers, learning pipelines, and portable configurations.
          </p>
          <span class="text-link">Browse GitHub ↗</span></a
        >
        <a href="https://www.modelscope.cn/datasets/chiyodamomo01/NeuralPOM"
          ><span class="eyebrow">03 / REFERENCE DATA</span>
          <h3>Meet the <br />reference ocean.</h3>
          <p>Reference data and dataset documentation on ModelScope.</p>
          <span class="text-link">Visit the dataset ↗</span></a
        >
      </div>
    </section>
    <div class="home-colophon">
      <span>NEURALPOM <i> / </i> A RESEARCH FIELD GUIDE</span
      ><a href="/guide/installation">Begin with installation →</a>
    </div>
  </div>
</template>
