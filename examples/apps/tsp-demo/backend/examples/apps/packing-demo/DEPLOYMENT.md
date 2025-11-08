# Deployment Guide

## Quick Deploy Options

### 1. **Vercel** (Recommended - Easiest)
**Best for:** Quick deployment, automatic builds, free tier

**Steps:**
1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com) and sign up/login
3. Click "New Project" and import your GitHub repository
4. Configure:
   - **Root Directory:** `apps/playground-web`
   - **Build Command:** `cd ../.. && npm install && npm run build --workspace=apps/playground-web`
   - **Output Directory:** `apps/playground-web/dist`
   - **Install Command:** `npm install`
5. Deploy!

**Note:** You may need to create a `vercel.json` in the root for monorepo support.

### 2. **Netlify**
**Best for:** Easy deployment, good free tier

**Steps:**
1. Push to GitHub
2. Go to [netlify.com](https://netlify.com) and sign up
3. Click "New site from Git"
4. Configure:
   - **Base directory:** `apps/playground-web`
   - **Build command:** `cd ../.. && npm install && npm run build --workspace=apps/playground-web`
   - **Publish directory:** `apps/playground-web/dist`
5. Deploy!

### 3. **Cloudflare Pages**
**Best for:** Fast CDN, free tier

**Steps:**
1. Push to GitHub
2. Go to [pages.cloudflare.com](https://pages.cloudflare.com)
3. Connect repository
4. Configure build settings similar to Netlify

### 4. **GitHub Pages**
**Best for:** Free, simple static hosting

**Steps:**
1. Build locally: `cd apps/playground-web && npm run build`
2. Push `dist` folder to `gh-pages` branch
3. Enable GitHub Pages in repository settings

## Pre-Deployment Checklist

1. **Build the core package first:**
   ```bash
   cd packages/core
   npm run build
   ```

2. **Build the web app:**
   ```bash
   cd apps/playground-web
   npm run build
   ```

3. **Test the build locally:**
   ```bash
   npm run preview
   ```

## Recommended: Vercel Configuration

Create `vercel.json` in the root directory:

```json
{
  "buildCommand": "cd apps/playground-web && npm run build",
  "outputDirectory": "apps/playground-web/dist",
  "installCommand": "npm install",
  "framework": "vite"
}
```

Or configure in Vercel dashboard:
- **Framework Preset:** Vite
- **Root Directory:** `apps/playground-web`
- **Build Command:** `npm run build`
- **Output Directory:** `dist`

## Environment Variables

If you need any environment variables, add them in your deployment platform's settings.

## Custom Domain

All platforms support custom domains:
- Vercel: Add domain in project settings
- Netlify: Add domain in site settings
- Cloudflare: Add domain in Pages settings

