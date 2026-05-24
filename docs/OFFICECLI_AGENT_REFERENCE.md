# OfficeCLI Agent Reference

> Compact reference for AI agents. Feed this into the system prompt.
> For full details: `officecli help` or `officecli help <format> <element>`.

## Quick Reference

```bash
CREATE:  officecli create <file>.{pptx|docx|xlsx}
ADD:     officecli add <file> <path> --type <element> --prop key=value
SET:     officecli set <file> <path> --prop key=value
GET:     officecli get <file> <path> --json
QUERY:   officecli query <file> "selector" --json
VIEW:    officecli view <file> outline|text|annotated|stats|issues|html|screenshot
WATCH:   officecli watch <file>                    # Live preview → :26315
REMOVE:  officecli remove <file> <path>
MOVE:    officecli move <file> <path> --to <parent> [--index N]
MERGE:   officecli merge template.docx out.docx '{"key":"val"}'
DUMP:    officecli dump <file> -o blueprint.json   # Learn from template
BATCH:   officecli batch <file> --input ops.json    # Atomic multi-command
VALIDATE: officecli validate <file>
```

## Help System (use this, don't guess)

```bash
officecli help                              # All commands
officecli help pptx                         # All PPTX elements
officecli help pptx set shape               # Properties with examples
officecli help pptx set shape --json        # Machine-readable schema
```

## Format Aliases
`word`=`docx`  `excel`=`xlsx`  `ppt`/`powerpoint`=`pptx`

## Three-Layer Strategy
Always prefer higher layers:

| Layer | Purpose | Example |
|-------|---------|---------|
| L1 Read | View content | `officecli view deck.pptx outline` |
| L2 DOM | Structured ops | `officecli add deck.pptx / --type slide` |
| L3 Raw | XML fallback | `officecli raw deck.pptx '/slide[1]'` |

## Path Syntax

1-based indexing, element local names (not XPath):

```
/                         # Document root
/slide[1]                 # First slide
/slide[1]/shape[2]        # Second shape on first slide
/body/p[3]                # Third paragraph (Word)
/Sheet1/A1                # Cell A1 (Excel)
/Sheet1                   # First sheet (Excel)
```

## Common Patterns

```bash
# PowerPoint — Create presentation with content
officecli create deck.pptx
officecli add deck.pptx / --type slide --prop title="Q4 Report" --prop background=1A1A2E
officecli add deck.pptx '/slide[1]' --type shape --prop text="Revenue: $4.2M" --prop x=2cm --prop y=5cm
officecli add deck.pptx '/slide[1]' --type picture --prop src=/path/to/chart.png --prop x=10cm --prop y=5cm
officecli add deck.pptx '/slide[1]' --type 3dmodel --prop src=/path/to/model.glb

# Word — Create document with paragraphs, tables, images
officecli create report.docx
officecli add report.docx /body --type paragraph --prop text="Executive Summary"
officecli add report.docx /body --type paragraph --prop text="Details..." --prop style=Heading1
officecli add report.docx /body --type table --prop rows=3 --prop cols=4
officecli set report.docx '/body/tbl[1]/row[1]/cell[1]' --prop text="Category"

# Excel — Create spreadsheet with formulas, charts, pivot tables
officecli create budget.xlsx
officecli add budget.xlsx / --type sheet --prop name="Q1"
officecli set budget.xlsx '/Q1/A1' --prop value="Revenue"
officecli set budget.xlsx '/Q1/B1' --prop value="=SUM(B2:B10)"
officecli add budget.xlsx '/Q1' --type chart --prop type=bar --prop source='A1:B10'
officecli add budget.xlsx '/Q1' --type pivottable \
  --prop source='Data!A1:E1000' --prop rows='Category' --prop values='Amount:sum'
```

## Units & Colors

| Type | Accepted | Examples |
|------|----------|----------|
| Dimensions | cm, in, pt, px, EMU | `2cm`, `1in`, `72pt`, `914400` |
| Colors | Hex, named, RGB, theme | `#FF0000`, `FF0000`, `red`, `accent1` |
| Font sizes | Number or pt | `14`, `14pt` |
| Spacing | pt, cm, in, multiplier | `12pt`, `0.5cm`, `1.5x`, `150%` |

## Structured Output

Always use `--json` for deterministic output:

```bash
officecli get deck.pptx '/slide[1]' --json
# → {"tag":"slide","path":"/slide[1]","attributes":{"title":"Q4 Report"}}

officecli query report.docx "paragraph[style=Heading1]" --json
# → [{"tag":"paragraph","path":"/body/p[1]","attributes":{...}}, ...]
```

## Error Recovery

```bash
# Wrong path → get suggestion
officecli get report.docx /body/p[99] --json
# → {"success":false,"error":{"code":"not_found","suggestion":"Valid range: 1-8"}}

# Self-correct by listing children
officecli get report.docx /body --depth 1 --json
```

## Output Directory
All files in: `/workspace/office-output/`

## Live Preview
```bash
officecli watch <file>    # → http://localhost:26315
# Every add/set/remove auto-refreshes the browser
# Excel: inline cell editing + drag-to-reposition charts
```
