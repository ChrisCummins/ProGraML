genrule(
    name = "main",
    srcs = glob([
        "**/*.tex",
        "eushield-normal.pdf",
        "eushield-normal.sty",
        "infthesis.cls",
        "thesis-shield.pdf",
        'eushield.sty',
        'refs.bib',
    ]),
    outs = ["main.pdf"],
    cmd = "$(location //tools:autotex) thesis/thesis.tex $@",
    tools = ["//tools:autotex"],
)
