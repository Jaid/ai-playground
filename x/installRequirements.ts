import * as path from 'forward-slash-path'
import fs from 'fs-extra'
import {globby} from 'globby'

import {$venv, $verbose} from 'lib/execa.js'

const venvFolder = path.join(`.venv`)
const venvFolderExists = await fs.pathExists(venvFolder)
if (!venvFolderExists) {
  console.error(`Please create a .venv folder first`)
  process.exit(1)
}
const hasPackage = async (packageName: string) => {
  const packageFolder = path.join(venvFolder, `Lib`, `site-packages`, packageName)
  return fs.pathExists(packageFolder)
}
await $venv`${$venv.pip} install --upgrade pip`
await $verbose`${$venv.pip} install setuptools wheel packaging`
const requirementsFiles = await globby([`requirements.txt`, `src/*/requirements.txt`])
const requirementsArgs = requirementsFiles.reduce<Array<string>>((acc, file) => {
  acc.push(`--requirement`)
  acc.push(file)
  return acc
}, [])
await $verbose`${$venv.pip} install --upgrade ${requirementsArgs}`
const hasFlashAttn = await hasPackage(`flash-attn`)
if (!hasFlashAttn) {
  await $verbose({
    env: {
      MAX_JOBS: `4`,
    },
  })`${$venv.pip} install flash-attn --no-build-isolation`
}
const hasTensorRt = await hasPackage(`tensorrt`)
if (!hasTensorRt) {
  await $verbose`${$venv.pip} install --no-cache-dir nvidia-cudnn-cu11`
  await $verbose`${$venv.pip} install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.3.0.post12.dev1`
  // await $verbose`${pipBase} uninstall -y nvidia-cudnn-cu11`
}
