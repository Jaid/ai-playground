import {$, type Execa$} from 'execa'
import * as path from 'forward-slash-path'

interface ExtendedExca extends Execa$ {
  venv: PythonExeca
  verbose: Execa$
}
interface PythonExeca extends Execa$ {
  pip: Array<string>
  python: Array<string>
}
const execaShell = $({}) as ExtendedExca
export const $verbose = $({
  stderr: `inherit`,
  stdout: `inherit`,
  verbose: true,
})
const venvFolder = path.join(import.meta.dirname, `..`, `.venv`, process.platform === `win32` ? `Scripts` : `bin`)
const pythonExecutableFile = path.join(venvFolder, process.platform === `win32` ? `python.exe` : `python`)
export const $venv = $verbose({
  preferLocal: true,
  localDir: path.join(import.meta.dirname, `..`, `.venv`, process.platform === `win32` ? `Scripts` : `bin`),
}) as PythonExeca
execaShell.verbose = $verbose
$venv.python = [
  pythonExecutableFile,
]
$venv.pip = [
  pythonExecutableFile,
  `-m`,
  `pip`,
  `--require-virtualenv`,
  `--disable-pip-version-check`,
]
execaShell.venv = $venv

export {execaShell as $}
