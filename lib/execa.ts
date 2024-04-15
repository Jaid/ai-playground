import {$, type Execa$} from 'execa'

interface VerboseExeca extends Execa$ {
  verbose: Execa$
}
const execaShell = $({}) as VerboseExeca
export const $verbose = $({
  stderr: `inherit`,
  stdin: `ignore`,
  stdout: `inherit`,
  verbose: true,
})
execaShell.verbose = $verbose

export {execaShell as $}
