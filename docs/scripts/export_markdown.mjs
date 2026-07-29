import { cp, mkdir, readdir, rm, stat } from "node:fs/promises";
import path from "node:path";

const sourceRoot = path.resolve("out/llms.mdx/docs");
const targetRoot = path.resolve("out/docs");

async function exportTree(directory) {
  for (const entry of await readdir(directory)) {
    const source = path.join(directory, entry);
    const info = await stat(source);
    if (info.isDirectory()) {
      await exportTree(source);
      continue;
    }

    const relative = path.relative(sourceRoot, source);
    const target = path.join(targetRoot, `${relative}.md`);
    await mkdir(path.dirname(target), { recursive: true });
    await cp(source, target);
  }
}

await exportTree(sourceRoot);
await rm(path.resolve("out/llms.mdx"), { recursive: true });
