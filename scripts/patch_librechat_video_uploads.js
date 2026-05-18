const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const alphaRavisDefaultEndpoint = 'LangGraph Agent';

const targets = [
  {
    path: '/app/packages/api/dist/index.js',
    blockStart: 'function encodeAndFormatVideos(req, files, params, getStrategyFunctions) {',
    insertAfter: '        const { provider, endpoint } = params;\n',
    insertion:
      `        const alphaRavisVideoEndpoints = new Set((process.env.ALPHARAVIS_LIBRECHAT_VIDEO_ENDPOINTS || '${alphaRavisDefaultEndpoint}').split(',').map((value) => value.trim()).filter(Boolean));\n` +
      '        const alphaRavisVideoEndpoint = alphaRavisVideoEndpoints.has(endpoint);\n',
    guard:
      "            if (!file.type.startsWith('video/') || !librechatDataProvider.isDocumentSupportedProvider(provider)) {",
    guardReplacement:
      "            if (!file.type.startsWith('video/') || (!librechatDataProvider.isDocumentSupportedProvider(provider) && !alphaRavisVideoEndpoint)) {",
    openRouter:
      '            else if (provider === agents.Providers.OPENROUTER) {',
    openRouterReplacement:
      '            else if (provider === agents.Providers.OPENROUTER || alphaRavisVideoEndpoint) {',
  },
  {
    path: '/app/packages/api/src/files/encode/video.ts',
    blockStart: 'export async function encodeAndFormatVideos(',
    insertAfter: '  const { provider, endpoint } = params;\n',
    insertion:
      `  const alphaRavisVideoEndpoints = new Set((process.env.ALPHARAVIS_LIBRECHAT_VIDEO_ENDPOINTS || '${alphaRavisDefaultEndpoint}').split(',').map((value) => value.trim()).filter(Boolean));\n` +
      '  const alphaRavisVideoEndpoint = alphaRavisVideoEndpoints.has(endpoint);\n',
    guard:
      "    if (!file.type.startsWith('video/') || !isDocumentSupportedProvider(provider)) {",
    guardReplacement:
      "    if (!file.type.startsWith('video/') || (!isDocumentSupportedProvider(provider) && !alphaRavisVideoEndpoint)) {",
    openRouter:
      '    } else if (provider === Providers.OPENROUTER) {',
    openRouterReplacement:
      '    } else if (provider === Providers.OPENROUTER || alphaRavisVideoEndpoint) {',
  },
];

function patchTarget(target) {
  if (!fs.existsSync(target.path)) {
    console.log(`[AlphaRavis LibreChat patch] skipped missing ${target.path}`);
    return;
  }

  let text = fs.readFileSync(target.path, 'utf8');
  const blockIndex = text.indexOf(target.blockStart);
  if (blockIndex < 0) {
    throw new Error(`Could not find video encoder block in ${target.path}`);
  }

  const before = text.slice(0, blockIndex);
  let block = text.slice(blockIndex);

  if (!block.includes('alphaRavisVideoEndpoint')) {
    block = block.replace(target.insertAfter, target.insertAfter + target.insertion);
  }

  if (block.includes(target.guard)) {
    block = block.replace(target.guard, target.guardReplacement);
  }

  if (block.includes(target.openRouter)) {
    block = block.replace(target.openRouter, target.openRouterReplacement);
  }

  const updated = before + block;
  if (updated !== text) {
    fs.writeFileSync(target.path, updated);
    console.log(`[AlphaRavis LibreChat patch] patched ${target.path}`);
  } else {
    console.log(`[AlphaRavis LibreChat patch] already patched ${target.path}`);
  }
}

for (const target of targets) {
  patchTarget(target);
}

function replaceOnce(text, search, replacement, label) {
  if (text.includes(replacement)) {
    return { text, changed: false, alreadyPatched: true };
  }
  if (!text.includes(search)) {
    throw new Error(`Could not find ${label}`);
  }
  return { text: text.replace(search, replacement), changed: true, alreadyPatched: false };
}

function patchPromptFormatter() {
  const file = '/app/api/app/clients/prompts/formatMessages.js';
  if (!fs.existsSync(file)) {
    console.log(`[AlphaRavis LibreChat patch] skipped missing ${file}`);
    return;
  }

  const search =
    '  const { image_urls } = message;\n' +
    "  if (Array.isArray(image_urls) && image_urls.length > 0 && role === 'user') {\n" +
    '    return formatVisionMessage({\n' +
    '      message: formattedMessage,\n' +
    '      image_urls: message.image_urls,\n' +
    '      endpoint,\n' +
    '    });\n' +
    '  }\n';
  const replacement =
    '  const media_parts = [\n' +
    '    ...(Array.isArray(message.image_urls) ? message.image_urls : []),\n' +
    '    ...(Array.isArray(message.videos) ? message.videos : []),\n' +
    '    ...(Array.isArray(message.audios) ? message.audios : []),\n' +
    '  ];\n' +
    "  if (media_parts.length > 0 && role === 'user') {\n" +
    '    return formatVisionMessage({\n' +
    '      message: formattedMessage,\n' +
    '      image_urls: media_parts,\n' +
    '      endpoint,\n' +
    '    });\n' +
    '  }\n';

  let text = fs.readFileSync(file, 'utf8');
  const result = replaceOnce(text, search, replacement, 'formatMessage media parts');
  text = result.text;

  if (result.changed) {
    fs.writeFileSync(file, text);
    console.log(`[AlphaRavis LibreChat patch] patched ${file}`);
  } else {
    console.log(`[AlphaRavis LibreChat patch] already patched ${file}`);
  }
}

function patchResponsesMediaConverters() {
  const targets = [
    {
      path: '/app/node_modules/@librechat/agents/dist/cjs/llm/openai/utils/index.cjs',
      quote: "'",
    },
    {
      path: '/app/node_modules/@langchain/openai/dist/chat_models.cjs',
      quote: '"',
    },
    {
      path: '/app/node_modules/@langchain/openai/dist/chat_models.js',
      quote: '"',
    },
  ];

  for (const target of targets) {
    if (!fs.existsSync(target.path)) {
      console.log(`[AlphaRavis LibreChat patch] skipped missing ${target.path}`);
      continue;
    }

    const q = target.quote;
    const imageBlock =
      `                if (item.type === ${q}image_url${q}) {\n` +
      '                    return {\n' +
      `                        type: ${q}input_image${q},\n` +
      '                        image_url: typeof item.image_url === ' +
      `${q}string${q}\n` +
      '                            ? item.image_url\n' +
      '                            : item.image_url.url,\n' +
      '                        detail: typeof item.image_url === ' +
      `${q}string${q}\n` +
      `                            ? ${q}auto${q}\n` +
      '                            : item.image_url.detail,\n' +
      '                    };\n' +
      '                }\n';
    const mediaBlock =
      imageBlock +
      `                if (item.type === ${q}video_url${q}) {\n` +
      '                    return {\n' +
      `                        type: ${q}input_video${q},\n` +
      '                        video_url: typeof item.video_url === ' +
      `${q}string${q}\n` +
      '                            ? item.video_url\n' +
      '                            : item.video_url.url,\n' +
      '                    };\n' +
      '                }\n';
    const allowBlock =
      `                if (item.type === ${q}input_text${q} ||\n` +
      `                    item.type === ${q}input_image${q} ||\n` +
      `                    item.type === ${q}input_file${q}) {\n`;
    const allowReplacement =
      `                if (item.type === ${q}input_text${q} ||\n` +
      `                    item.type === ${q}input_image${q} ||\n` +
      `                    item.type === ${q}input_video${q} ||\n` +
      `                    item.type === ${q}input_file${q}) {\n`;

    let text = fs.readFileSync(target.path, 'utf8');
    let changed = false;

    let result = replaceOnce(
      text,
      imageBlock,
      mediaBlock,
      `Responses video_url conversion in ${target.path}`,
    );
    text = result.text;
    changed ||= result.changed;

    result = replaceOnce(
      text,
      allowBlock,
      allowReplacement,
      `Responses input_video passthrough in ${target.path}`,
    );
    text = result.text;
    changed ||= result.changed;

    if (changed) {
      fs.writeFileSync(target.path, text);
      console.log(`[AlphaRavis LibreChat patch] patched ${target.path}`);
    } else {
      console.log(`[AlphaRavis LibreChat patch] already patched ${target.path}`);
    }
  }
}

function patchClientSource() {
  const sourceTargets = [
    {
      path: '/app/client/src/components/Chat/Input/Files/AttachFileMenu.tsx',
      replacements: [
        {
          label: 'AttachFileMenu provider capability',
          search:
            '        isAzureWithResponsesApi\n' +
            '      ) {\n',
          replacement:
            '        isAzureWithResponsesApi ||\n' +
            `        currentProvider === '${alphaRavisDefaultEndpoint}'\n` +
            '      ) {\n',
        },
        {
          label: 'AttachFileMenu video file type',
          search:
            '            if (currentProvider === Providers.GOOGLE || currentProvider === Providers.OPENROUTER) {\n',
          replacement:
            '            if (\n' +
            '              currentProvider === Providers.GOOGLE ||\n' +
            '              currentProvider === Providers.OPENROUTER ||\n' +
            `              currentProvider === '${alphaRavisDefaultEndpoint}'\n` +
            '            ) {\n',
        },
      ],
    },
    {
      path: '/app/client/src/components/Chat/Input/Files/DragDropModal.tsx',
      replacements: [
        {
          label: 'DragDropModal provider capability',
          search:
            '      isAzureWithResponsesApi\n' +
            '    ) {\n',
          replacement:
            '      isAzureWithResponsesApi ||\n' +
            `      currentProvider === '${alphaRavisDefaultEndpoint}'\n` +
            '    ) {\n',
        },
        {
          label: 'DragDropModal video validation',
          search:
            '        currentProvider === EModelEndpoint.google || currentProvider === Providers.OPENROUTER;\n',
          replacement:
            '        currentProvider === EModelEndpoint.google ||\n' +
            '        currentProvider === Providers.OPENROUTER ||\n' +
            `        currentProvider === '${alphaRavisDefaultEndpoint}';\n`,
        },
      ],
    },
  ];

  for (const target of sourceTargets) {
    if (!fs.existsSync(target.path)) {
      console.log(`[AlphaRavis LibreChat patch] skipped missing ${target.path}`);
      continue;
    }
    let text = fs.readFileSync(target.path, 'utf8');
    let changed = false;
    for (const replacement of target.replacements) {
      const result = replaceOnce(text, replacement.search, replacement.replacement, replacement.label);
      text = result.text;
      changed ||= result.changed;
    }
    if (changed) {
      fs.writeFileSync(target.path, text);
      console.log(`[AlphaRavis LibreChat patch] patched ${target.path}`);
    } else {
      console.log(`[AlphaRavis LibreChat patch] already patched ${target.path}`);
    }
  }
}

function patchClientDist() {
  const assetsDir = '/app/client/dist/assets';
  if (!fs.existsSync(assetsDir)) {
    console.log(`[AlphaRavis LibreChat patch] skipped missing ${assetsDir}`);
    return;
  }

  const distFiles = fs
    .readdirSync(assetsDir)
    .filter((name) => name.startsWith('index.') && name.endsWith('.js'))
    .map((name) => path.join(assetsDir, name));

  for (const file of distFiles) {
    let text = fs.readFileSync(file, 'utf8');
    let changed = false;
    const replacements = [
      {
        label: 'dist drag/drop provider capability',
        search: 'if(Wu(m)||Wu(n)||r){const r=n===$u.google||n===Bu.OPENROUTER',
        replacement:
          `if(Wu(m)||Wu(n)||r||n==="${alphaRavisDefaultEndpoint}"){const r=n===$u.google||n===Bu.OPENROUTER||n==="${alphaRavisDefaultEndpoint}"`,
      },
      {
        label: 'dist attach menu provider capability',
        search:
          'return Wu(r)||Wu(s)||o?n.push({label:p("com_ui_upload_provider"),onClick:()=>{y(void 0);let t="image_document";s===Bu.GOOGLE||s===Bu.OPENROUTER?t="image_document_video_audio"',
        replacement:
          `return Wu(r)||Wu(s)||o||s==="${alphaRavisDefaultEndpoint}"?n.push({label:p("com_ui_upload_provider"),onClick:()=>{y(void 0);let t="image_document";s===Bu.GOOGLE||s===Bu.OPENROUTER||s==="${alphaRavisDefaultEndpoint}"?t="image_document_video_audio"`,
      },
    ];

    for (const replacement of replacements) {
      const result = replaceOnce(text, replacement.search, replacement.replacement, replacement.label);
      text = result.text;
      changed ||= result.changed;
    }

    if (text.includes(alphaRavisDefaultEndpoint)) {
      fs.writeFileSync(`${file}.gz`, zlib.gzipSync(text, { level: 9 }));
      fs.writeFileSync(`${file}.br`, zlib.brotliCompressSync(text));
    }

    if (changed) {
      fs.writeFileSync(file, text);
      console.log(`[AlphaRavis LibreChat patch] patched ${file}`);
    } else {
      console.log(`[AlphaRavis LibreChat patch] already patched ${file}`);
    }
  }
}

patchPromptFormatter();
patchResponsesMediaConverters();
patchClientSource();
patchClientDist();

function writeCompressedSiblings(file, text) {
  if (!fs.existsSync(file)) {
    return;
  }
  fs.writeFileSync(file, text);
  fs.writeFileSync(`${file}.gz`, zlib.gzipSync(text, { level: 9 }));
  fs.writeFileSync(`${file}.br`, zlib.brotliCompressSync(text));
}

function patchServiceWorkerCache() {
  const registerPath = '/app/client/dist/registerSW.js';
  const swPath = '/app/client/dist/sw.js';

  const unregisterScript = `
if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    try {
      const registrations = await navigator.serviceWorker.getRegistrations();
      await Promise.all(registrations.map((registration) => registration.unregister()));
      if ('caches' in window) {
        const keys = await caches.keys();
        await Promise.all(keys.map((key) => caches.delete(key)));
      }
    } catch (error) {
      console.warn('[AlphaRavis] failed to clear LibreChat service worker cache', error);
    }
  });
}
`.trim();

  const clearingServiceWorker = `
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((key) => caches.delete(key)));
    await self.clients.claim();
    const clients = await self.clients.matchAll({ type: 'window' });
    for (const client of clients) {
      client.navigate(client.url);
    }
    await self.registration.unregister();
  })());
});
`.trim();

  writeCompressedSiblings(registerPath, unregisterScript);
  writeCompressedSiblings(swPath, clearingServiceWorker);
  console.log('[AlphaRavis LibreChat patch] disabled LibreChat service worker cache');
}

patchServiceWorkerCache();
