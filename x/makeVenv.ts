import {$} from 'execa'
import fs from 'fs-extra'
import * as path from 'forward-slash-path'

const pythonMajor = 3
const pythonMinor = 11
const venvFolder = path.join(`.venv`)
const venvFolderAlreadyExists = await fs.pathExists(venvFolder)
if (venvFolderAlreadyExists) {
  await fs.remove(venvFolder)
}
const $verbose = $({
  stderr: `inherit`,
  stdin: `ignore`,
  stdout: `inherit`,
})
await $verbose`py -${pythonMajor}.${pythonMinor} -m venv ${venvFolder}`
const gitignoreFile = path.join(venvFolder, `.gitignore`)
await fs.outputFile(gitignoreFile, `*`)
