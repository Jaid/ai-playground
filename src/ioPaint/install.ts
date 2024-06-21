import * as path from 'forward-slash-path'
import fs from 'fs-extra'

import {$venv, $verbose} from 'lib/execa.js'

// #!/usr/bin/env bash
// set -e

// pushd ./web_app
// rm -r -f dist
// npm run build
// popd
// rm -r -f ./iopaint/web_app
// cp -r web_app/dist ./iopaint/web_app

// rm -r -f dist
// python3 setup.py sdist bdist_wheel

const slug = `Sanster/IOPaint`
const branch = `iopaint-1.2.2`
const rootFolder = path.join(import.meta.dirname, `..`, `..`)
const repoFolder = path.join(rootFolder, `temp`, `git`, `ioPaint`)
await fs.remove(repoFolder)
const repoFolderExists = await fs.pathExists(repoFolder)
const $verboseInRepo = $verbose({
  cwd: repoFolder,
})
const $venvInRepo = $venv({
  cwd: repoFolder,
})
try {
  if (repoFolderExists) {
    await $verboseInRepo`git pull`
    await $verboseInRepo`git log -1 --oneline`
  } else {
    await $verbose`git clone --branch ${branch} git@github.com:${slug}.git ${repoFolder}`
    await $verboseInRepo`git log -1 --oneline`
  }
} catch (error) {
  await $verboseInRepo`git status`
  throw error
}
await fs.remove(path.join(repoFolder, `web_app/dist`))
await $verbose({
  cwd: path.join(repoFolder, `web_app`),
})`npm install`
await $verbose({
  cwd: path.join(repoFolder, `web_app`),
})`npm run build`
await fs.remove(path.join(repoFolder, `iopaint/web_app`))
await fs.copy(path.join(repoFolder, `web_app/dist`), path.join(repoFolder, `iopaint/web_app`))
await fs.remove(path.join(rootFolder, `dist`))
await $venvInRepo`${$venv.python} setup.py sdist bdist_wheel`
