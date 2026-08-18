<template>
  <!-- Page number / count on every slide -->
  <div
    v-if="$slidev.nav.currentPage > 1"
    class="pointer-events-none absolute bottom-2 right-3 text-xs opacity-50 z-10"
  >
    {{ $slidev.nav.currentPage }} / {{ $slidev.nav.total }}
  </div>
</template>
