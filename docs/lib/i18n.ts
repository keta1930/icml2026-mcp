import { defineI18n } from "fumadocs-core/i18n";
import { defineI18nUI, type Translations } from "fumadocs-ui/i18n";

export const languages = ["en", "zh"] as const;
export type Language = (typeof languages)[number];
export const defaultLanguage: Language = "en";

export const i18n = defineI18n({
  languages: [...languages],
  defaultLanguage,
  parser: "dir",
  fallbackLanguage: null,
  hideLocale: "never",
});

const zhTranslations = {
  displayName: "简体中文",
  "Ask AI(AI chat button)": "询问 AI",
  "Back to Home(404 page)": "返回首页",
  "Choose a language(language switcher)": "选择语言",
  "Choose a language(language switcher)(aria-label)": "选择语言",
  "Close Banner(banner)(aria-label)": "关闭横幅",
  "Close Search(search dialog)(aria-label)": "关闭搜索",
  "Close Sidebar(aria-label)": "关闭侧边栏",
  "Close Sidebar(sidebar)(aria-label)": "关闭侧边栏",
  "Collapse Sidebar(sidebar)(aria-label)": "收起侧边栏",
  "Copied Text(code block)(aria-label)": "已复制文本",
  "Copy Anchor Link(heading anchor)(aria-label)": "复制标题链接",
  "Copy Link(accordion)(aria-label)": "复制链接",
  "Copy Markdown(page actions)": "复制 Markdown",
  "Copy Text(code block)(aria-label)": "复制文本",
  "Dark(theme switcher)(aria-label)": "深色",
  "Default(type table)": "默认值",
  "Edit on GitHub(edit page)": "在 GitHub 编辑",
  "Hide Sidebar(sidebar)": "隐藏侧边栏",
  "Last updated on(page footer)": "最后更新于",
  "Layout Tab(layout tab trigger)": "布局标签",
  "Light(theme switcher)(aria-label)": "浅色",
  "Next Page(pagination)": "下一页",
  "No Headings(table of contents)": "无标题",
  "No results found(search dialog)": "未找到结果",
  "On this page(table of contents)": "本页内容",
  "Open Search(search trigger)(aria-label)": "打开搜索",
  "Open Sidebar(sidebar)(aria-label)": "打开侧边栏",
  "Open in ChatGPT(page actions)": "在 ChatGPT 中打开",
  "Open in Claude(page actions)": "在 Claude 中打开",
  "Open in Cursor(page actions)": "在 Cursor 中打开",
  "Open in GitHub(page actions)": "在 GitHub 中打开",
  "Open in Scira AI(page actions)": "在 Scira AI 中打开",
  "Open(page actions)": "打开",
  "Page Not Found(404 page)": "页面未找到",
  "Parameters(type table)": "参数",
  "Previous Page(pagination)": "上一页",
  "Prop(type table)": "属性",
  "Read {url}, I want to ask questions about it.(page actions)":
    "请阅读 {url}，我想就其内容提问。",
  "Returns(type table)": "返回值",
  "Search(search dialog)": "搜索",
  "Search(search trigger)": "搜索",
  "Show Sidebar(sidebar)": "显示侧边栏",
  "System(theme switcher)(aria-label)": "跟随系统",
  "Table of Contents(inline table of contents)": "目录",
  "The page you are looking for might have been removed, had its name changed, or is temporarily unavailable.(404 page)":
    "你访问的页面可能已被删除、更名或暂时不可用。",
  "Toggle Menu(mobile menu)(aria-label)": "切换菜单",
  "Toggle Theme(theme switcher)(aria-label)": "切换主题",
  "Type(type table)": "类型",
  "View as Markdown(page actions)": "查看 Markdown",
} satisfies Partial<Translations>;

export const uiI18n = defineI18nUI(i18n, {
  en: { displayName: "English" },
  zh: zhTranslations,
});

export function isLanguage(value: string): value is Language {
  return languages.some((language) => language === value);
}

export function requireLanguage(value: string): Language {
  if (!isLanguage(value)) throw new Error(`Unsupported documentation locale: ${value}`);
  return value;
}

export function htmlLanguage(language: Language): string {
  return language === "zh" ? "zh-CN" : "en";
}
