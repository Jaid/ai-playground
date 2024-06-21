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
await $verbose`${$venv.pip} install --upgrade --requirement ${path.join(import.meta.dirname, `requirements.txt`)}`
