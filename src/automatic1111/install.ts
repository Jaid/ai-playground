import {fileURLToPath} from 'node:url'

import {$} from 'execa'
import fs from 'fs-extra'
import path from 'zeug/path'

const automatic1111Slug = `automatic1111/stable-diffusion-webui`
const forgeSlug = `lllyasviel/stable-diffusion-webui-forge`
const automatic1111Branch = `master`
const dirname = path.dirname(fileURLToPath(import.meta.url))
const rootFolder = path.join(dirname, `..`, `..`)
const venvFolder = path.join(rootFolder, `.venv`)
const $verbose = $({
  stderr: `inherit`,
  stdin: `ignore`,
  stdout: `inherit`,
})
const repoFolder = path.join(dirname, `git`, `automatic1111`)
const gitArgs = [
  `-C`,
  repoFolder,
]
const gitCommandArgs = []
try {
  await fs.emptyDir(repoFolder)
  await $verbose`git ${gitArgs} clone ${gitCommandArgs} --branch ${automatic1111Branch} git@github.com:${automatic1111Slug}.git ${repoFolder}`
  await $verbose`git ${gitArgs} log ${gitCommandArgs} -1 --oneline`
  await $verbose`git ${gitArgs} remote ${gitCommandArgs} add forge git@github.com:${forgeSlug}.git`
  await $verbose`git ${gitArgs} branch ${gitCommandArgs} lllyasviel/main`
  await $verbose`git ${gitArgs} checkout ${gitCommandArgs} lllyasviel/main`
  await $verbose`git ${gitArgs} fetch ${gitCommandArgs} forge`
  await $verbose`git ${gitArgs} branch ${gitCommandArgs} -u forge/main`
  await $verbose`git ${gitArgs} pull ${gitCommandArgs} --strategy-option theirs`
  await $verbose`git ${gitArgs} add ${gitCommandArgs} .`
} catch (error) {
  await $verbose`git ${gitArgs} status ${gitCommandArgs}`
  throw error
}
