RealityCheck AI
Break your AI model before reality does.

Computer vision models can look great on a benchmark and still miss something important when the lighting changes, an object is partly hidden, or a frame gets blurry. RealityCheck AI is an experiment in making those weak spots easier to find, reproduce, and understand.

I'm building it for the Meta Global AI Developer Hackathon (2026). The first version focuses on one simple workflow: upload an image, pick an object, generate a few realistic stress tests, and see how an object detector responds.

This is a work in progress, not a finished robustness certification tool.
What I'm building
1. Understand the scene. Run a baseline object detector; use SAM 3.1 to segment the object we want to test.
2. Choose meaningful tests. Use Muse Spark to propose challenges for the scene, such as partial occlusion or poor visibility.
3. Change one thing at a time. Apply reproducible blur, lighting, and object-local transformations to create test images.
4. Run the detector again. Compare predictions before and after each change.
5. Show the evidence. Display the images, predictions, configuration, and results so the experiment can be repeated.
The key idea is test → measure → learn where the model struggles, rather than just generate altered images.
Why this matters
Robustness testing often comes down to assembling datasets and hand-picking edge cases. I want to make that loop faster and more accessible, while keeping the measurements honest. A change in confidence is not automatically proof of a failure; verified detection metrics require reliable labels.
Current status
🚧 Active hackathon build (October 27–November 4, 2026). This README describes the intended MVP. Features below aren't guaranteed to work until their implementation and tests land in this repository.
MVP checklist
- [ ] Upload and display an image
- [ ] Run an object detector and visualize bounding boxes
- [ ] Integrate Meta's SAM 3.1 segmentation
- [ ] Use Muse Spark to propose structured stress tests
- [ ] Apply deterministic transformations with recorded seeds
- [ ] Re-run detection and compare outcomes
- [ ] Produce a visual, reproducible evaluation report
- [ ] Deploy an accessible live demo
Planned stack
- Python, PyTorch, OpenCV, Pillow — inference and transformations
- Meta Model API — Muse Spark reasoning and SAM 3.1 segmentation
- Gradio — interactive prototype
- pytest — unit and integration tests
The architecture and dependencies may change as the MVP takes shape.
Getting started
The application entry point and installation commands will be added once the first integrated workflow is committed. In the meantime, see the commit history for what's already implemented. An API key may be required for Meta-hosted models; credentials must never be committed.
How results will be reported
For labeled test images, I plan to report measures such as recall or missed detections using checked ground truth. For unlabeled uploads, the app will report changes in predictions and confidence as consistency signals, not verified failures. Generated tests will include their parameters and seed where applicable.
Built during the hackathon
The repository history will distinguish new hackathon work from any external dependencies, pretrained models, datasets, and third-party components. Those components retain their own licenses and usage terms.
Contributing
Bug reports, reproducible failures, and thoughtful PRs are welcome. If you find a surprising example, include the input source or license, model version, transformation settings, and expected versus actual behavior. Please don't upload private images or material you don't have permission to share.
License and citation
The original code in this repository is licensed under Apache License 2.0. See LICENSE and NOTICE. If you redistribute the software or a derivative work, you must meet Apache 2.0's attribution and notice-preservation conditions.

