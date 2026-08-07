from nntp.datasets.TaskRegistry import TaskRegistry
from .RobertYang2019 import generate_trials, get_default_hp
from .RobertYang2019Generator import getTaskGenerator

# Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T., & Wang, X.-J. (2019). Task representations in neural networks trained to perform many cognitive tasks. Nature Neuroscience, 22(2), 297–306. https://doi.org/10.1038/s41593-018-0310-2
CITATION = "Guangyu Robert Yang, et al. 2019"
decoder_path = "RobertYang2019.RobertYang2019Decoder:ring_decoder"
evaluator_path = "RobertYang2019.RobertYang2019Evaluator:build_evaluator"

get_task_generator = lambda task_name: getTaskGenerator(
    task_name, generate_trials, get_default_hp("all")
)
TaskRegistry.register(
    "fdgo-ry",
    "Go",
    CITATION,
    "A single stimulus is randomly shown in either modality 1 or 2, and the response should be made in the direction of the stimulus. The stimulus appears before the fixation cue goes off.",
    get_task_generator("fdgo"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "reactgo-ry",
    "RT Go",
    CITATION,
    "A single stimulus is randomly shown in either modality 1 or 2, and the response should be made in the direction of the stimulus. The fixation input never goes off, and the network should respond as soon as the stimulus appears.",
    get_task_generator("reactgo"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "delaygo-ry",
    "Dly Go",
    CITATION,
    "A single stimulus is randomly shown in either modality 1 or 2, and the response should be made in the direction of the stimulus. A stimulus appears briefly and is followed by a delay period until the fixation cue goes off.",
    get_task_generator("delaygo"),
    decoder_path,
    evaluator_path,
)

TaskRegistry.register(
    "fdanti-ry",
    "Anti",
    CITATION,
    "A single stimulus is randomly shown in either modality 1 or 2, and the response should be made in the opposite direction of the stimulus. The stimulus appears before the fixation cue goes off.",
    get_task_generator("fdanti"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "reactanti-ry",
    "RT Anti",
    CITATION,
    "A single stimulus is randomly shown in either modality 1 or 2, and the response should be made in the opposite direction of the stimulus. The fixation input never goes off, and the network should respond as soon as the stimulus appears.",
    get_task_generator("reactanti"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "delayanti-ry",
    "Dly Anti",
    CITATION,
    "A single stimulus is randomly shown in either modality 1 or 2, and the response should be made in the opposite direction of the stimulus. A stimulus appears briefly and is followed by a delay period until the fixation cue goes off.",
    get_task_generator("delayanti"),
    decoder_path,
    evaluator_path,
)

TaskRegistry.register(
    "dm1-ry",
    "DM 1",
    CITATION,
    "In each trial, two stimuli are shown simultaneously and are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. The two stimuli only appear in modality 1. The correct response should be made to the direction of the stronger stimulus (the stimulus with higher amplitude).",
    get_task_generator("dm1"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "dm2-ry",
    "DM 2",
    CITATION,
    "In each trial, two stimuli are shown simultaneously and are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. The two stimuli only appear in modality 2. The correct response should be made to the direction of the stronger stimulus (the stimulus with higher amplitude).",
    get_task_generator("dm2"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "contextdm1-ry",
    "Ctx DM 1",
    CITATION,
    "In each trial, two stimuli are shown simultaneously and are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. Each stimulus appears in both modality 1 and 2. Information from modality 2 should be ignored, and the correct response should be made to the stronger stimulus in modality 1.",
    get_task_generator("contextdm1"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "contextdm2-ry",
    "Ctx DM 2",
    CITATION,
    "In each trial, two stimuli are shown simultaneously and are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. Each stimulus appears in both modality 1 and 2. Information from modality 1 should be ignored, and the correct response should be made to the stronger stimulus in modality 2.",
    get_task_generator("contextdm2"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "multidm-ry",
    "MultSen DM",
    CITATION,
    "In each trial, two stimuli are shown simultaneously and are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. Each stimulus appears in both modality 1 and 2. The correct response should be made to the stimulus that has a stronger combined strength in modalities 1 and 2",
    get_task_generator("multidm"),
    decoder_path,
    evaluator_path,
)

TaskRegistry.register(
    "delaydm1-ry",
    "Dly DM 1",
    CITATION,
    "In each trial, two stimuli are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. The two stimuli only appear in modality 1. The two stimuli are separated in time. The correct response should be made to the direction of the stronger stimulus (the stimulus with higher amplitude).",
    get_task_generator("delaydm1"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "delaydm2-ry",
    "Dly DM 2",
    CITATION,
    "In each trial, two stimuli are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. The two stimuli only appear in modality 2. The two stimuli are separated in time. The correct response should be made to the direction of the stronger stimulus (the stimulus with higher amplitude).",
    get_task_generator("delaydm2"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "contextdelaydm1-ry",
    "Ctx Dly DM 1",
    CITATION,
    "In each trial, two stimuli are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. Each stimulus appears in both modality 1 and 2. The two stimuli are separated in time. Information from modality 2 should be ignored, and the correct response should be made to the stronger stimulus in modality 1.",
    get_task_generator("contextdelaydm1"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "contextdelaydm2-ry",
    "Ctx Dly DM 2",
    CITATION,
    "In each trial, two stimuli are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. Each stimulus appears in both modality 1 and 2. The two stimuli are separated in time. Information from modality 1 should be ignored, and the correct response should be made to the stronger stimulus in modality 2.",
    get_task_generator("contextdelaydm2"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "multidelaydm-ry",
    "MultSen Dly DM",
    CITATION,
    "In each trial, two stimuli are presented till the end of the trial. Stimulus 1 is drawn randomly between 0 and 360°, while stimulus 2 is drawn uniformly between 90 and 270° away from stimulus 1. Each stimulus appears in both modality 1 and 2. The two stimuli are separated in time. The correct response should be made to the stimulus that has a stronger combined strength in modalities 1 and 2",
    get_task_generator("multidelaydm"),
    decoder_path,
    evaluator_path,
)

TaskRegistry.register(
    "dmsgo-ry",
    "DMS",
    CITATION,
    "Two stimuli are presented consecutively and separated by a delay period. Each stimulus can appear in either modality 1 or 2. The network response depends on whether or not the two stimuli are 'matched'. Two stimuli are matched if they point toward the same direction, regardless of their modalities. The network should respond toward the direction of the second stimulus if the two stimuli are matched and maintain fixation otherwise.",
    get_task_generator("dmsgo"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "dmsnogo-ry",
    "DNMS",
    CITATION,
    "Two stimuli are presented consecutively and separated by a delay period. Each stimulus can appear in either modality 1 or 2. The network response depends on whether or not the two stimuli are 'matched'. Two stimuli are matched if they point toward the same direction, regardless of their modalities. The network should respond only if the two stimuli are not matched, that is, a non-match, and fixate when it is a match.",
    get_task_generator("dmsnogo"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "dmcgo-ry",
    "DMC",
    CITATION,
    "Two stimuli are presented consecutively and separated by a delay period. Each stimulus can appear in either modality 1 or 2. The network response depends on whether or not the two stimuli are 'matched'. Two stimuli are matched if their directions belong to the same category. The first category ranges from 0 to 180°, while the rest from 180 to 360° belong to the second category. The network should respond toward the direction of the second stimulus if the two stimuli are matched and maintain fixation otherwise.",
    get_task_generator("dmcgo"),
    decoder_path,
    evaluator_path,
)
TaskRegistry.register(
    "dmcnogo-ry",
    "DNMC",
    CITATION,
    "Two stimuli are presented consecutively and separated by a delay period. Each stimulus can appear in either modality 1 or 2. The network response depends on whether or not the two stimuli are 'matched'. Two stimuli are matched if their directions belong to the same category. The first category ranges from 0 to 180°, while the rest from 180 to 360° belong to the second category. The network should respond only if the two stimuli are not matched, that is, a non-match, and fixate when it is a match.",
    get_task_generator("dmcnogo"),
    decoder_path,
    evaluator_path,
)

# TaskRegistry.register("oic-ry", "1IC", CITATION, "", get_task_generator("oic"))
# TaskRegistry.register(
#     "dmc-ry",
#     "DMC",
#     CITATION,
#     "Two stimuli are presented consecutively and separated by a delay period. Each stimulus can appear in either modality 1 or 2. The network response depends on whether or not the two stimuli are 'matched'.",
#     get_task_generator("dmc"),
#     decoder_path,
#     evaluator_path,
#
# )
