"""Build the working-paper PDF from its Markdown source with Pandoc and Typst."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import pypandoc
import typst

root = Path(__file__).resolve().parent
content = pypandoc.convert_file(root/"manuscript.md","typst",format="gfm")
style = '''
#set document(title: "Testing RGB Feature Transfer for Pose-Free Feed-Forward Gaussian Reconstruction from Events")
#set page(paper: "a4", margin: (x: 22mm, y: 19mm), numbering: "1")
#set text(size: 10pt, lang: "en")
#set par(justify: true, leading: 0.68em)
#set block(spacing: 0.65em)
#show heading.where(level: 1): set text(size: 18pt)
#show heading.where(level: 2): set text(size: 13pt)
#show heading.where(level: 3): set text(size: 11pt)
#show table: set text(size: 8pt)
'''
# Typst resolves local images relative to the input document.
with NamedTemporaryFile(suffix=".typ",dir=root,delete=False) as tmp:
    source = Path(tmp.name)
try:
    source.write_text(style+content)
    typst.compile(str(source),output=str(root/"manuscript.pdf"),root=str(root))
finally:
    source.unlink(missing_ok=True)
