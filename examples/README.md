# Example inputs

These are **AI-generated**: no real people appear in them. They are copied into ComfyUI's `input/` folder on first
start, so the bundled workflows run straight away.

| File | What it is | Sound |
|---|---|---|
| `drivers/talking_kitchen.mp4` | a woman in a kitchen talks to the camera, 832x480, 5 s | "Hi everyone! Today I'm going to show you the easiest pancakes you will ever make." |
| `drivers/talking_bench.mp4` | a man on a park bench talks to the camera, 832x480, 5 s | "Honestly, I never thought I would see a dog ride a skateboard. But here we are!" |
| `drivers/presenter.mp4` | a presenter in front of a white wall, 832x480, 5 s | "Welcome back. In the next five minutes, you will learn three tricks that will change how you cook." |
| `drivers/dance_studio.mp4` | a man dances in a studio, full body, 480x832, 5 s | music |
| `drivers/wave_street.mp4` | a woman in a raincoat walks up a street and waves, full body, 480x832, 5 s | street ambience |
| `characters/*.jpg` | knight, robot, wizard, pirate, fox, gingerbread man: full-body characters on a plain backdrop, 768x1024 | - |

Made with [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (22B distilled, bf16, text to video with audio,
8 steps) from text prompts. Lightricks claims no rights in outputs (LTX-2 Community License, section 5). The files
are released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/); please keep them labelled as
AI-generated.
