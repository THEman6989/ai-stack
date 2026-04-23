# 1. Basis-Image mit Node.js 20
FROM node:20-alpine AS base

# 2. Abhängigkeiten installieren
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Kopiere Paketmanager-Dateien aus dem Build-Kontext (Submodul)
COPY package.json yarn.lock* ./
RUN yarn install --frozen-lockfile

# 3. Build-Stufe
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN yarn build

# 4. Runner-Stufe (Produktions-Image)
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

USER nextjs

EXPOSE 3000
ENV PORT 3000

CMD ["yarn", "start"]
