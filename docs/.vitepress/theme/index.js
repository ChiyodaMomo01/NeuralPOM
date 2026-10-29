import DefaultTheme from "vitepress/theme";
import ResearchHome from "./components/ResearchHome.vue";
import DensityExplorer from "./components/DensityExplorer.vue";
import "./custom.css";

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component("ResearchHome", ResearchHome);
    app.component("DensityExplorer", DensityExplorer);
  },
};
