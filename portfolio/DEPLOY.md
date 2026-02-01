# Deployment Guide (GitHub Pages)

## 1. Prepare for GitHub Pages
Since GitHub Pages hosts your site at `https://username.github.io/repo-name/`, we need to tell Vite about this base path.

1.  Open `vite.config.js`.
2.  Add `base: '/portfolio/',` (replace `portfolio` with your actual repo name if different).

```javascript
export default defineConfig({
  base: '/portfolio/', // <--- ADD THIS LINE
  plugins: [react(), tailwindcss()],
})
```

## 2. Install gh-pages
Run this command to install the deployment tool:
```bash
npm install gh-pages --save-dev
```

## 3. Update package.json
Add these two scripts to your `package.json` under `"scripts"`:

```json
"scripts": {
  "predeploy": "npm run build",
  "deploy": "gh-pages -d dist",
  ...
}
```

## 4. Push to GitHub
1.  Create a new repository on GitHub (e.g., `portfolio`).
2.  Initialize git and push your code:
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/portfolio.git
    git push -u origin main
    ```

## 5. Deploy 🚀
Run the deploy script:
```bash
npm run deploy
```

Your site will be live at `https://YOUR_USERNAME.github.io/portfolio/`!
