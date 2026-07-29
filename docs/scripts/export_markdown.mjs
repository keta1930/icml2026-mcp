import { cp, mkdir, readdir, rm, stat } from "node:fs/promises";
import path from "node:path";

const languages = ["en", "zh"];

async function countFiles(directory) {
  let count = 0;
  for (const entry of await readdir(directory)) {
    const target = path.join(directory, entry);
    const info = await stat(target);
    count += info.isDirectory() ? await countFiles(target) : 1;
  }
  return count;
}

async function exportTree(directory, sourceRoot, targetRoot) {
  let copied = 0;
  for (const entry of await readdir(directory)) {
    const source = path.join(directory, entry);
    const info = await stat(source);
    if (info.isDirectory()) {
      copied += await exportTree(source, sourceRoot, targetRoot);
      continue;
    }

    const relative = path.relative(sourceRoot, source);
    const target = path.join(targetRoot, `${relative}.md`);
    await mkdir(path.dirname(target), { recursive: true });
    await cp(source, target);
    copied += 1;
  }
  return copied;
}

const trees = await Promise.all(
  languages.map(async (language) => {
    const exportRoot = path.resolve("out", language, "llms.mdx");
    const sourceRoot = path.join(exportRoot, "docs");
    return {
      language,
      exportRoot,
      sourceRoot,
      targetRoot: path.resolve("out", language, "docs"),
      count: await countFiles(sourceRoot),
    };
  }),
);
const counts = new Set(trees.map((tree) => tree.count));
if (counts.size !== 1 || trees[0].count === 0) {
  throw new Error(
    `Markdown source trees must be non-empty and equal: ${trees
      .map((tree) => `${tree.language}=${tree.count}`)
      .join(", ")}`,
  );
}

for (const tree of trees) {
  const copied = await exportTree(tree.sourceRoot, tree.sourceRoot, tree.targetRoot);
  if (copied !== tree.count) {
    throw new Error(`incomplete Markdown export for locale ${tree.language}`);
  }
}
for (const tree of trees) {
  await rm(tree.exportRoot, { recursive: true });
  process.stdout.write(
    `exported Markdown: ${tree.language} (${tree.count + 1} pages)\n`,
  );
}
