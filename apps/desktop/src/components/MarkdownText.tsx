import type { ReactNode } from "react";

type MarkdownTextProps = {
  children: string;
  className?: string;
};

function safeHref(value: string): string | null {
  const href = value.trim();
  return /^(?:https?:\/\/|mailto:)/i.test(href) ? href : null;
}

function inlineContent(value: string, keyPrefix = "inline"): ReactNode[] {
  const nodes: ReactNode[] = [];
  let text = "";
  let index = 0;

  const flush = () => {
    if (text) {
      nodes.push(text);
      text = "";
    }
  };

  while (index < value.length) {
    if (value[index] === "\\") {
      const next = value[index + 1];
      if (next && "\\`*_[]()".includes(next)) {
        text += next;
        index += 2;
        continue;
      }
    }

    if (value[index] === "`") {
      const end = value.indexOf("`", index + 1);
      if (end > index + 1) {
        flush();
        nodes.push(
          <code key={`${keyPrefix}-code-${index}`} className="rounded px-1.5 py-0.5 font-mono text-[0.9em]" style={{ background: "var(--bg-3)", color: "var(--accent)" }}>
            {value.slice(index + 1, end)}
          </code>,
        );
        index = end + 1;
        continue;
      }
    }

    const link = value.slice(index).match(/^\[([^\]]+)\]\(([^\s)]+)(?:\s+['\"][^'\"]*['\"])?\)/);
    if (link) {
      flush();
      const href = safeHref(link[2]);
      nodes.push(
        href ? (
          <a key={`${keyPrefix}-link-${index}`} href={href} target="_blank" rel="noreferrer" className="underline decoration-[var(--accent-ring)] underline-offset-2" style={{ color: "var(--accent)" }}>
            {inlineContent(link[1], `${keyPrefix}-link-${index}`)}
          </a>
        ) : (
          <span key={`${keyPrefix}-link-${index}`}>{inlineContent(link[1], `${keyPrefix}-link-${index}`)}</span>
        ),
      );
      index += link[0].length;
      continue;
    }

    const strong = value.slice(index).match(/^(\*\*|__)(?=\S)(.+?)(?<=\S)\1/);
    if (strong) {
      flush();
      nodes.push(<strong key={`${keyPrefix}-strong-${index}`}>{inlineContent(strong[2], `${keyPrefix}-strong-${index}`)}</strong>);
      index += strong[0].length;
      continue;
    }

    const emphasis = value.slice(index).match(/^(\*|_)(?=\S)(.+?)(?<=\S)\1/);
    if (emphasis) {
      flush();
      nodes.push(<em key={`${keyPrefix}-em-${index}`}>{inlineContent(emphasis[2], `${keyPrefix}-em-${index}`)}</em>);
      index += emphasis[0].length;
      continue;
    }

    text += value[index];
    index += 1;
  }

  flush();
  return nodes;
}

type Block =
  | { type: "paragraph"; lines: string[] }
  | { type: "heading"; level: number; text: string }
  | { type: "ul"; items: string[] }
  | { type: "ol"; items: string[] }
  | { type: "table"; headers: string[]; rows: string[][]; alignments: Array<"left" | "center" | "right"> }
  | { type: "code"; language: string; text: string };

function splitTableRow(line: string): string[] | null {
  const trimmed = line.trim();
  if (!trimmed.includes("|")) return null;
  const source = trimmed.startsWith("|") ? trimmed.slice(1) : trimmed;
  // A trailing pipe only acts as the row's right edge when it is not itself
  // escaped (preceded by an even run of backslashes). With an odd run it is
  // part of the last cell, e.g. `a \|` renders an escaped pipe, so the escape
  // parser below must still see it.
  let backslashes = 0;
  let cursor = source.length - 1;
  while (cursor > 0 && source[cursor - 1] === "\\") {
    backslashes += 1;
    cursor -= 1;
  }
  const content =
    source.endsWith("|") && backslashes % 2 === 0 ? source.slice(0, -1) : source;
  const cells: string[] = [];
  let cell = "";
  let index = 0;
  while (index < content.length) {
    const character = content[index];
    if (character === "\\") {
      const next = content[index + 1];
      // A backslash only escapes a cell separator (or another backslash).
      // Otherwise it stays verbatim so inline formatting is not corrupted.
      if (next === "|" || next === "\\") {
        cell += next;
        index += 2;
      } else {
        cell += character;
        index += 1;
      }
    } else if (character === "|") {
      cells.push(cell.trim());
      cell = "";
      index += 1;
    } else {
      cell += character;
      index += 1;
    }
  }
  cells.push(cell.trim());
  return cells;
}

function tableAlignments(line: string, count: number): Array<"left" | "center" | "right"> | null {
  const cells = splitTableRow(line);
  if (!cells || cells.length !== count || cells.some((cell) => !/^:?-{3,}:?$/.test(cell))) return null;
  return cells.map((cell) => {
    const left = cell.startsWith(":");
    const right = cell.endsWith(":");
    return left && right ? "center" : right ? "right" : "left";
  });
}

function parseBlocks(value: string): Block[] {
  const lines = value.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let paragraph: string[] = [];
  let list: Extract<Block, { type: "ul" | "ol" }> | null = null;
  let index = 0;

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ type: "paragraph", lines: paragraph });
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list) {
      blocks.push(list);
      list = null;
    }
  };

  while (index < lines.length) {
    const line = lines[index];
    const fence = line.match(/^\s*```([^`]*)\s*$/);
    if (fence) {
      flushParagraph();
      flushList();
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push({ type: "code", language: fence[1].trim(), text: codeLines.join("\n") });
      continue;
    }

    const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2] });
      index += 1;
      continue;
    }

    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const type = unordered ? "ul" : "ol";
      if (!list || list.type !== type) {
        flushList();
        list = { type, items: [] };
      }
      if (list) list.items.push((unordered || ordered)?.[1] || "");
      index += 1;
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      index += 1;
      continue;
    }

    // A table is recognized only when the following line is a valid Markdown
    // separator. This prevents ordinary prose containing a pipe from being
    // reformatted as a table.
    const tableHeaders = splitTableRow(line);
    const alignments = index + 1 < lines.length
      ? tableAlignments(lines[index + 1], tableHeaders?.length || 0)
      : null;
    if (tableHeaders && alignments) {
      flushParagraph();
      flushList();
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length) {
        const row = splitTableRow(lines[index]);
        if (!row || row.length !== tableHeaders.length) break;
        rows.push(row);
        index += 1;
      }
      blocks.push({ type: "table", headers: tableHeaders, rows, alignments });
      continue;
    }

    flushList();
    paragraph.push(line);
    index += 1;
  }

  flushParagraph();
  flushList();
  return blocks;
}

export function MarkdownText({ children, className = "" }: MarkdownTextProps) {
  const blocks = parseBlocks(children);
  return (
    <div className={`markdown-text space-y-2 ${className}`}>
      {blocks.map((block, index) => {
        if (block.type === "code") {
          return (
            <pre key={`block-${index}`} className="overflow-x-auto rounded-lg p-3 text-[12px] leading-relaxed" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
              <code className="font-mono" data-language={block.language || undefined}>{block.text}</code>
            </pre>
          );
        }
        if (block.type === "heading") {
          const Heading = `h${Math.min(block.level, 6)}` as keyof JSX.IntrinsicElements;
          return <Heading key={`block-${index}`} className="font-semibold leading-tight" style={{ fontSize: `${Math.max(0.9, 1.15 - block.level * 0.06)}em` }}>{inlineContent(block.text, `block-${index}`)}</Heading>;
        }
        if (block.type === "ul" || block.type === "ol") {
          const List = block.type;
          return <List key={`block-${index}`} className="ml-5 space-y-1" style={{ listStyleType: block.type === "ul" ? "disc" : "decimal" }}>{block.items.map((item, itemIndex) => <li key={`${index}-${itemIndex}`}>{inlineContent(item, `block-${index}-item-${itemIndex}`)}</li>)}</List>;
        }
        if (block.type === "table") {
          return (
            <div key={`block-${index}`} className="overflow-x-auto rounded-lg" style={{ border: "1px solid var(--bg-4)" }}>
              <table className="w-full min-w-max border-collapse text-left text-[12px]">
                <thead style={{ background: "var(--bg-2)" }}>
                  <tr>
                    {block.headers.map((header, cellIndex) => (
                      <th
                        key={`${index}-header-${cellIndex}`}
                        className="px-3 py-2 font-semibold"
                        style={{ borderBottom: "1px solid var(--bg-4)", textAlign: block.alignments[cellIndex] }}
                      >
                        {inlineContent(header, `block-${index}-header-${cellIndex}`)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, rowIndex) => (
                    <tr key={`${index}-row-${rowIndex}`} style={{ borderBottom: rowIndex < block.rows.length - 1 ? "1px solid var(--bg-4)" : undefined }}>
                      {block.headers.map((_header, cellIndex) => (
                        <td key={`${index}-${rowIndex}-${cellIndex}`} className="px-3 py-2 align-top" style={{ textAlign: block.alignments[cellIndex] }}>
                          {inlineContent(row[cellIndex] || "", `block-${index}-row-${rowIndex}-cell-${cellIndex}`)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return <p key={`block-${index}`} className="whitespace-pre-wrap break-words">{block.lines.map((line, lineIndex) => <span key={`${index}-${lineIndex}`}>{inlineContent(line, `block-${index}-line-${lineIndex}`)}{lineIndex < block.lines.length - 1 && <br />}</span>)}</p>;
      })}
    </div>
  );
}
