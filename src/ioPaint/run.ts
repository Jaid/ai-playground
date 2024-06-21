import * as path from 'forward-slash-path'

import {$venv} from 'lib/execa.js'

const rootFolder = path.join(import.meta.dirname, `..`, `..`)
const repoFolder = path.join(rootFolder, `temp`, `git`, `ioPaint`)
const entry = path.join(repoFolder, `main.py`)
await $venv({
  cwd: repoFolder,
  env: {
    VITE_BACKEND: `http://localhost:8080`,
  },
})`${$venv.python} ${entry} start --disable-nsfw-checker --device cuda --quality 100 --output-dir C:/Users/jaid/Pictures/out/ioPaint --model mat --enable-remove-bg --remove-bg-model briaai/RMBG-1.4 --enable-interactive-seg  --interactive-seg-model vit_h`
