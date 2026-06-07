# Frontend image: Vite dev server for the React SPA.
FROM node:20-slim

WORKDIR /app

# Install deps first for better layer caching.
COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev"]
