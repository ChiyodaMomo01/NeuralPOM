<script setup>
import { ref, computed } from "vue";
const retention = ref(75);
const squared = computed(() => ((retention.value / 100) ** 2 * 100).toFixed(2));
</script>
<template>
  <div class="density-explorer">
    <div class="explorer-heading">
      <span class="eyebrow">EXPLORE THE PROJECTION</span
      ><button type="button" @click="retention = 75">
        Reset to paper setting
      </button>
    </div>
    <div class="explorer-body">
      <svg
        viewBox="0 0 300 235"
        role="img"
        :aria-label="`Density-normal amplitude retained: ${retention} percent. The orthogonal component is unchanged.`"
      >
        <defs>
          <marker
            id="vector-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M0 0L10 5L0 10Z" fill="currentColor" />
          </marker>
        </defs>
        <g fill="none" stroke="currentColor">
          <path d="M45 195H270M45 195V20" opacity=".25" />
          <path d="M45 195L225 45" stroke-dasharray="4 5" opacity=".4" />
          <path
            :d="`M225 195V${195 - retention * 1.5}`"
            stroke-width="2"
            stroke-dasharray="3 4"
            class="normal-vector"
          />
          <path
            :d="`M45 195L225 ${195 - retention * 1.5}`"
            stroke-width="3"
            marker-end="url(#vector-arrow)"
          />
          <path d="M45 195H225" stroke-width="3" class="tangent-vector" />
        </g>
        <circle
          :cy="195 - retention * 1.5"
          cx="225"
          r="5"
          fill="currentColor"
        />
        <text x="75" y="222">Orthogonal component</text>
        <text transform="translate(20 168) rotate(-90)">
          Density-normal component
        </text>
      </svg>
      <div class="explorer-controls">
        <label for="retention"
          >Density-normal amplitude <strong>{{ retention }}%</strong></label
        ><input
          id="retention"
          v-model.number="retention"
          type="range"
          min="0"
          max="100"
          step="1"
        />
        <div class="range-labels">
          <span>0% · removed</span><span>100% · retained</span>
        </div>
        <div class="explorer-result">
          <strong>{{ squared }}<small>%</small></strong
          ><span>of squared projection norm retained</span>
        </div>
        <p>
          The orthogonal component stays fixed. At the paper setting, the
          squared projection norm decreases by 43.75%.
        </p>
      </div>
    </div>
    <p class="explorer-caption">
      Schematic in projection coordinates, before the final paired limiter. The
      slider illustrates the linear projection identity; it does not run an
      ocean forecast.
    </p>
  </div>
</template>
