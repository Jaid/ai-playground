import os from 'node:os'

import {$} from 'execa'
import fs from 'fs-extra'
import {globby} from 'globby'
import path from 'forward-slash-path'

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
const pythonExecutableFile = os.platform() === `win32` ? path.join(venvFolder, `Scripts`, `python.exe`) : path.join(venvFolder, `bin`, `python`)
const pipBase = [
  pythonExecutableFile,
  `-m`,
  `pip`,
  `--require-virtualenv`,
  `--disable-pip-version-check`,
]
const $verbose = $({
  stderr: `inherit`,
  stdin: `ignore`,
  stdout: `inherit`,
})
await $verbose`${pipBase} install --upgrade pip`
await $verbose`${pipBase} install setuptools wheel packaging`
const requirementsFiles = await globby([`requirements.txt`, `src/*/requirements.txt`])
const requirementsArgs = requirementsFiles.reduce<string[]>((acc, file) => {
  acc.push(`--requirement`)
  acc.push(file)
  return acc
}, [])
await $verbose`${pipBase} install --upgrade ${requirementsArgs}`
const hasFlashAttn = await hasPackage(`flash-attn`)
if (!hasFlashAttn) {
  await $verbose({
    env: {
      MAX_JOBS: `4`,
    },
  })`${pipBase} install flash-attn --no-build-isolation`
}
const hasTensorRt = await hasPackage(`tensorrt`)
if (!hasTensorRt) {
  await $verbose`${pipBase} install --no-cache-dir nvidia-cudnn-cu11`
  await $verbose`${pipBase} install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.3.0.post12.dev1`
  // await $verbose`${pipBase} uninstall -y nvidia-cudnn-cu11`
}
