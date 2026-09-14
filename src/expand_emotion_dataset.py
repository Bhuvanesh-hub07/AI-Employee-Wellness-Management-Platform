"""
Expand and balance the workplace emotion training dataset.

The script:
1. Loads the original emotion dataset.
2. Combines original examples with additional examples.
3. Removes duplicate texts.
4. Keeps all original examples.
5. Adds unique examples until each emotion reaches the target size.
6. Validates class balance.
7. Shuffles the final dataset.
8. Saves the expanded dataset.
"""

from pathlib import Path
import random

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path("data/emotion_train.csv")
OUTPUT_FILE = Path("data/emotion_train_expanded.csv")

TARGET_PER_EMOTION = 100
RANDOM_SEED = 42

EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
]


# ---------------------------------------------------------------------
# Additional training examples
# ---------------------------------------------------------------------

additional_data = {
    "joy": [
        "I felt proud when I completed my assignment successfully.",
        "My team appreciated the effort I put into the project.",
        "I was excited to see my work producing good results.",
        "Receiving positive feedback made me feel happy.",
        "I enjoyed collaborating with my coworkers today.",
        "I felt encouraged after my manager praised my work.",
        "Completing the difficult task gave me a great sense of satisfaction.",
        "I was delighted when my project was approved.",
        "My successful presentation made me feel confident and happy.",
        "I felt grateful when my colleagues supported my idea.",
        "I was thrilled to achieve my weekly target.",
        "The successful meeting left me feeling positive.",
        "I enjoyed learning something new at work today.",
        "I felt motivated after hearing encouraging words from my team.",
        "My hard work finally paid off and I felt wonderful.",
        "I was pleased with the progress I made today.",
        "The appreciation from my coworkers made my day.",
        "I felt cheerful after solving the problem.",
        "I was happy to receive recognition for my contribution.",
        "Working on an interesting project makes me feel energetic.",
        "I felt satisfied after finishing everything on my task list.",
        "My team celebration made me feel appreciated.",
        "I was excited about the opportunities coming next.",
        "I felt positive after a productive day.",
        "The good news about my project made me smile.",
        "I enjoyed helping my colleague complete the task.",
        "I felt proud when my solution worked perfectly.",
        "My manager's encouraging message lifted my mood.",
        "I was delighted to see my team succeed.",
        "The successful result made all the effort worthwhile.",
        "I felt enthusiastic about starting the new assignment.",
        "I was pleased with how well the meeting went.",
        "My colleagues congratulated me and I felt happy.",
        "I felt hopeful about my career after the discussion.",
        "Finishing the project early made me feel great.",
        "I enjoyed the friendly atmosphere in the office.",
        "The positive response to my idea made me happy.",
        "I felt accomplished after reaching my goal.",
        "I was excited to share my achievement with my team.",
        "The recognition made me feel valued.",
        "I felt relaxed and happy after completing my responsibilities.",
        "I enjoyed today's teamwork.",
        "I felt lucky to work with such supportive colleagues.",
        "The successful demonstration made me proud.",
        "I was pleased that my efforts were noticed.",
        "I felt energized after receiving good news.",
        "The project milestone gave me a strong sense of achievement.",
        "I felt happy knowing that my contribution helped the team.",
        "I was grateful for the encouragement I received.",
        "The positive feedback increased my motivation.",
        "I felt joyful after completing the challenging work.",
        "I was excited when my proposal was accepted.",
        "My team's success made me genuinely happy.",
        "I felt confident after successfully handling the task.",
        "The appreciation from my manager made me smile.",
        "I enjoyed the sense of progress today.",
        "I felt proud of how much I accomplished.",
        "The successful outcome made me feel optimistic.",
        "I was happy to see my work making a difference.",
        "I felt encouraged by my team's support.",
        "The new opportunity made me feel excited.",
        "I enjoyed being part of the successful project.",
        "I felt positive after solving the customer's problem.",
        "The congratulations from my colleagues made me happy.",
        "I felt fulfilled after helping the team.",
        "I was excited to learn that my work had been selected.",
        "I felt proud when my manager highlighted my contribution.",
        "The achievement gave me a boost of confidence.",
        "I enjoyed working on the task with my teammates.",
        "I felt cheerful after receiving the good news.",
        "The team's appreciation made me feel valued.",
        "I was delighted by the successful project result.",
        "I felt motivated by the progress we made.",
        "I was happy that my suggestion was implemented.",
        "The positive meeting gave me renewed energy.",
        "I felt satisfied knowing I had done my best.",
        "I enjoyed the challenge and the final result.",
        "I felt proud of my improvement.",
        "The recognition encouraged me to keep improving.",
        "I was thrilled when the deadline was successfully met.",
        "I felt happy seeing everyone celebrate the achievement.",
        "The supportive environment made me feel comfortable.",
        "I felt optimistic about the next phase of the project.",
        "I was pleased with my performance today.",
        "The successful presentation made me feel accomplished.",
        "I felt grateful for the opportunity to contribute.",
        "I enjoyed the positive interaction with my coworkers.",
        "I felt excited about my professional growth.",
        "The good result made me feel confident.",
        "I was happy to receive constructive encouragement.",
        "I felt proud after overcoming the difficult problem.",
        "The team spirit made the work enjoyable.",
        "I felt enthusiastic about tomorrow's tasks.",
        "I was delighted that my effort was recognized.",
        "I felt fulfilled after completing the assignment.",
        "The positive outcome made me feel satisfied.",
        "I enjoyed seeing the project come together.",
        "I felt genuinely happy with my progress."
    ],

    "sadness": [
        "I felt disappointed when my work was ignored.",
        "I was sad after hearing that my project was cancelled.",
        "I feel emotionally drained after a difficult week.",
        "My lack of progress made me feel discouraged.",
        "I was upset that nobody noticed my effort.",
        "I felt lonely during the long workday.",
        "The negative feedback left me feeling down.",
        "I was disappointed when the opportunity was given to someone else.",
        "I feel tired and emotionally exhausted from work.",
        "I was sad to leave my supportive team.",
        "The failure of the project made me feel hopeless.",
        "I felt discouraged after making the same mistake again.",
        "I was unhappy with how the situation ended.",
        "I felt alone when my colleagues stopped communicating.",
        "The criticism made me lose confidence.",
        "I was disappointed by the lack of recognition.",
        "I felt miserable after receiving the rejection.",
        "The difficult workload left me emotionally exhausted.",
        "I was sad that my idea was dismissed.",
        "I felt low after an unsuccessful presentation.",
        "The stressful week made me feel emotionally tired.",
        "I was disappointed with my performance.",
        "I felt hurt when my contribution was overlooked.",
        "I became sad after hearing the disappointing news.",
        "I feel discouraged about my current progress.",
        "The rejection made me question my abilities.",
        "I felt unhappy with the way I was treated.",
        "I was saddened by the loss of an opportunity.",
        "The constant problems at work made me feel drained.",
        "I felt disappointed after failing to reach my goal.",
        "I was sad when my teammate left the organization.",
        "The lack of support made me feel alone.",
        "I felt emotionally exhausted after the meeting.",
        "I was discouraged by the repeated setbacks.",
        "The project delay made me feel disappointed.",
        "I felt unhappy about the outcome.",
        "I was sad that my effort did not produce results.",
        "The difficult situation affected my mood.",
        "I felt lonely working late without my team.",
        "I was disappointed by the unexpected decision.",
        "The negative result made me feel hopeless.",
        "I felt down after hearing criticism.",
        "I was sad about missing the opportunity.",
        "The long hours left me feeling exhausted.",
        "I felt hurt by the lack of appreciation.",
        "I was disappointed when my request was rejected.",
        "The failed attempt made me feel discouraged.",
        "I felt sad because my work was not valued.",
        "I was unhappy with the lack of progress.",
        "The situation made me feel emotionally tired.",
        "I felt disappointed after being left out.",
        "I was sad that my plans did not work.",
        "The constant pressure made me feel drained.",
        "I felt discouraged after the unsuccessful interview.",
        "I was unhappy with the final decision.",
        "The loss of support made me feel lonely.",
        "I felt sad when the project ended unsuccessfully.",
        "I was disappointed by the unexpected setback.",
        "The difficult conversation left me feeling low.",
        "I felt emotionally exhausted after handling the workload.",
        "I was sad about the poor outcome.",
        "The rejection made me feel discouraged.",
        "I felt hurt when my efforts were ignored.",
        "I was unhappy after receiving the negative result.",
        "The stressful environment affected my mood.",
        "I felt disappointed with the way things developed.",
        "I was sad when my colleague resigned.",
        "The repeated failures made me lose motivation.",
        "I felt lonely when nobody supported my idea.",
        "I was discouraged by the lack of improvement.",
        "The unexpected problem made me feel helpless.",
        "I felt sad after my presentation went badly.",
        "I was disappointed that my hard work did not matter.",
        "The long working hours made me emotionally tired.",
        "I felt unhappy about the situation.",
        "I was sad that my expectations were not met.",
        "The criticism left me feeling discouraged.",
        "I felt hurt by the unfair response.",
        "I was disappointed after missing the deadline.",
        "The stressful project made me feel exhausted.",
        "I felt sad about the conflict with my colleague.",
        "I was unhappy after the failed meeting.",
        "The lack of recognition made me feel low.",
        "I felt discouraged after receiving poor feedback.",
        "I was sad when my plans changed unexpectedly.",
        "The difficult workload made me feel emotionally drained.",
        "I felt disappointed by my team's decision.",
        "I was unhappy that my contribution was ignored.",
        "The repeated delays made me feel hopeless.",
        "I felt sad after losing an important opportunity.",
        "I was discouraged by the difficult circumstances.",
        "The negative outcome made me feel emotionally tired.",
        "I felt lonely during a challenging period.",
        "I was disappointed that things did not improve.",
        "The experience left me feeling sad and drained."
    ],

    "anger": [
        "I am frustrated by the unfair workload.",
        "I was angry when my deadline was changed without warning.",
        "My manager's unfair decision made me furious.",
        "I feel irritated by repeated interruptions.",
        "I am upset because my colleague took credit for my work.",
        "The unfair treatment made me angry.",
        "I was frustrated when the system failed again.",
        "I feel annoyed by constant unnecessary meetings.",
        "I became angry after being blamed for someone else's mistake.",
        "The repeated delays are making me furious.",
        "I am irritated by the lack of communication.",
        "I was angry about the unreasonable deadline.",
        "The unfair distribution of work frustrated me.",
        "I feel annoyed when my suggestions are ignored.",
        "I was furious when my effort was criticized unfairly.",
        "The constant changes are frustrating me.",
        "I am angry about being treated disrespectfully.",
        "I felt irritated after another unexpected task was assigned.",
        "The poor management made me frustrated.",
        "I am upset that nobody listened to my concerns.",
        "I was angry when my leave request was rejected without explanation.",
        "The unfair policy made me furious.",
        "I feel frustrated with the repeated technical problems.",
        "I became angry after being interrupted during my presentation.",
        "The disrespectful comment made me upset.",
        "I am irritated by people ignoring important instructions.",
        "I was furious about the careless mistake.",
        "The situation made me extremely frustrated.",
        "I feel angry when my workload keeps increasing.",
        "I was annoyed by the unnecessary criticism.",
        "The unfair feedback made me angry.",
        "I feel frustrated because responsibilities are not shared equally.",
        "I became angry when the meeting was changed at the last minute.",
        "The lack of accountability is frustrating.",
        "I am upset about the repeated broken promises.",
        "I was angry when my concerns were dismissed.",
        "The constant interruptions make me irritated.",
        "I feel furious about the unfair treatment.",
        "I became frustrated when my work was repeatedly delayed.",
        "The careless behavior made me angry.",
        "I am annoyed by the poor planning.",
        "I was upset because the rules were applied unfairly.",
        "The unreasonable expectations are frustrating.",
        "I feel angry about being given too much work.",
        "I was irritated by the negative attitude of my teammate.",
        "The repeated mistakes made me furious.",
        "I am frustrated because nobody takes responsibility.",
        "I became angry after being unfairly evaluated.",
        "The disrespect made me extremely upset.",
        "I feel annoyed when decisions are made without discussion.",
        "I was angry about the sudden workload increase.",
        "The unfair criticism frustrated me.",
        "I feel irritated by constant last-minute changes.",
        "I became furious when my contribution was ignored.",
        "The situation is making me increasingly angry.",
        "I was upset by the lack of fairness.",
        "The repeated technical failures are frustrating.",
        "I feel angry when people do not respect my time.",
        "I was irritated by the unnecessary delay.",
        "The unfair decision made me furious.",
        "I am frustrated with the way this project is being handled.",
        "I became angry when my request was ignored.",
        "The lack of support made me extremely annoyed.",
        "I feel upset about the disrespectful behavior.",
        "I was furious after receiving an unfair complaint.",
        "The constant pressure is making me angry.",
        "I am irritated by the repeated mistakes.",
        "I became frustrated when my work was rejected without explanation.",
        "The unfair workload distribution makes me angry.",
        "I feel annoyed by the lack of cooperation.",
        "I was angry when my schedule was changed suddenly.",
        "The situation made me lose my patience.",
        "I am frustrated by the poor communication.",
        "I became furious after being blamed unfairly.",
        "The repeated interruptions are extremely irritating.",
        "I feel angry about the lack of recognition.",
        "I was upset when my colleague ignored my concerns.",
        "The unfair treatment made me lose my temper.",
        "I am annoyed by unnecessary bureaucracy.",
        "I became frustrated with the slow response.",
        "The careless decision made me angry.",
        "I feel furious about the way the issue was handled.",
        "I was irritated by the constant noise.",
        "The repeated changes are driving me crazy.",
        "I am angry because my responsibilities keep increasing.",
        "I became upset after another unfair decision.",
        "The lack of respect is frustrating.",
        "I feel annoyed when people break agreements.",
        "I was furious about the poor treatment.",
        "The unfair situation made me extremely angry.",
        "I am frustrated with the constant problems.",
        "I became angry after my work was criticized unfairly.",
        "The repeated delays are making me lose patience.",
        "I feel irritated by the lack of support.",
        "I was upset about the unreasonable demand.",
        "The unfair behavior made me furious."
    ],

    "fear": [
        "I am worried about meeting my new responsibilities.",
        "I feel nervous about tomorrow's performance review.",
        "I am anxious about making a serious mistake.",
        "The upcoming deadline makes me nervous.",
        "I am afraid that I will not meet expectations.",
        "I feel worried about losing my job.",
        "The difficult presentation makes me anxious.",
        "I am nervous about speaking in front of management.",
        "I worry that my project will fail.",
        "I feel uneasy about the upcoming changes.",
        "I am afraid of making the wrong decision.",
        "The uncertain situation makes me nervous.",
        "I feel anxious about my future at work.",
        "I am worried that I will miss the deadline.",
        "The new responsibilities make me uncomfortable.",
        "I feel nervous before important meetings.",
        "I am afraid my performance will not be good enough.",
        "The possibility of failure worries me.",
        "I feel anxious when I have to present my ideas.",
        "I am concerned about the upcoming evaluation.",
        "The unexpected change made me nervous.",
        "I worry about disappointing my manager.",
        "I feel uneasy about working on the unfamiliar task.",
        "I am afraid that I might lose the opportunity.",
        "The uncertainty about the project makes me anxious.",
        "I feel nervous when deadlines are approaching.",
        "I am worried about handling the increased workload.",
        "The difficult conversation makes me uncomfortable.",
        "I fear that my mistakes will affect the team.",
        "I feel anxious about receiving feedback.",
        "I am nervous about starting the new role.",
        "The lack of information makes me worried.",
        "I am afraid that the project will be cancelled.",
        "I feel uneasy about the upcoming meeting.",
        "I worry about whether I can succeed.",
        "The possibility of failure makes me nervous.",
        "I feel anxious about the uncertain future.",
        "I am concerned about my ability to complete the task.",
        "The new environment makes me nervous.",
        "I fear making a bad impression.",
        "I feel worried about the difficult assignment.",
        "The unexpected responsibility makes me anxious.",
        "I am nervous about discussing my performance.",
        "I worry that my work will not be accepted.",
        "I feel uneasy when expectations are unclear.",
        "I am afraid of missing an important detail.",
        "The upcoming presentation makes me anxious.",
        "I feel nervous about working with a new team.",
        "I worry about making mistakes under pressure.",
        "The uncertainty makes me uncomfortable.",
        "I am afraid that I will disappoint everyone.",
        "I feel anxious about the approaching deadline.",
        "I am nervous about asking for help.",
        "I worry about the consequences of my decision.",
        "The difficult task makes me feel uneasy.",
        "I am concerned about my job security.",
        "I feel afraid when the situation becomes unpredictable.",
        "The upcoming interview makes me nervous.",
        "I worry that I may not be prepared.",
        "I feel anxious about the new expectations.",
        "I am afraid of failing the evaluation.",
        "The uncertain result makes me nervous.",
        "I feel worried about the consequences.",
        "I am uneasy about the changes in management.",
        "I fear that I will make another mistake.",
        "I feel nervous when important decisions are required.",
        "The possibility of conflict makes me anxious.",
        "I worry about whether my team will succeed.",
        "I am afraid that my performance will be judged negatively.",
        "The unfamiliar situation makes me uncomfortable.",
        "I feel anxious about taking responsibility.",
        "I am nervous about meeting the client.",
        "I worry that the workload is becoming unmanageable.",
        "The approaching review makes me uneasy.",
        "I feel afraid about the uncertain outcome.",
        "I am concerned that I may not finish on time.",
        "The new project makes me nervous.",
        "I worry about how others will react.",
        "I feel anxious before major presentations.",
        "I am afraid of making an embarrassing mistake.",
        "The lack of preparation makes me nervous.",
        "I feel uneasy about the difficult discussion.",
        "I worry that my efforts may fail.",
        "I am concerned about the upcoming decision.",
        "The unpredictable situation makes me anxious.",
        "I feel nervous about handling the responsibility alone.",
        "I am afraid that something could go wrong.",
        "The pressure makes me feel worried.",
        "I feel anxious about the next stage.",
        "I am nervous about the result.",
        "I worry about the future of the project.",
        "The uncertainty makes me feel afraid.",
        "I feel concerned about meeting expectations.",
        "I am nervous about what will happen next.",
        "The situation leaves me feeling uneasy."
    ],

    "surprise": [
        "I was surprised when my manager praised my work.",
        "I did not expect to receive such positive feedback.",
        "The unexpected promotion surprised me.",
        "I was amazed by the sudden announcement.",
        "I never expected my team to celebrate my achievement.",
        "The unexpected result caught me completely off guard.",
        "I was surprised when my proposal was accepted.",
        "The sudden change in plans surprised everyone.",
        "I was shocked by the unexpected decision.",
        "I did not expect to receive an award.",
        "The unexpected message surprised me.",
        "I was amazed when my colleague helped me unexpectedly.",
        "The sudden opportunity came as a surprise.",
        "I was surprised by how quickly the problem was solved.",
        "I never expected such a positive response.",
        "The unexpected success amazed me.",
        "I was surprised when the meeting was cancelled suddenly.",
        "The sudden announcement caught me by surprise.",
        "I did not expect my manager to support the idea.",
        "The unexpected recognition surprised me.",
        "I was amazed by the team's unexpected result.",
        "The sudden change surprised me.",
        "I was shocked when the deadline was moved.",
        "I never expected the project to succeed so quickly.",
        "The unexpected feedback caught me off guard.",
        "I was surprised by the unusual decision.",
        "The sudden opportunity was completely unexpected.",
        "I did not expect to be selected.",
        "The unexpected news surprised me.",
        "I was amazed by the result.",
        "The sudden improvement surprised everyone.",
        "I never thought the issue would be resolved so quickly.",
        "The unexpected invitation surprised me.",
        "I was shocked by the announcement.",
        "The unexpected support from my colleague surprised me.",
        "I did not expect such a quick response.",
        "The sudden success caught me by surprise.",
        "I was surprised by the positive outcome.",
        "The unexpected promotion came as a shock.",
        "I never expected to receive that opportunity.",
        "The sudden decision surprised the whole team.",
        "I was amazed by how things changed.",
        "The unexpected result left me surprised.",
        "I did not anticipate the announcement.",
        "The sudden change was surprising.",
        "I was shocked by the unexpected feedback.",
        "The unexpected recognition caught me off guard.",
        "I never imagined the project would finish early.",
        "The sudden invitation was unexpected.",
        "I was surprised when my idea was accepted.",
        "The unexpected result amazed me.",
        "I did not expect the manager to change the plan.",
        "The sudden announcement surprised me.",
        "I was shocked when the project was approved.",
        "The unexpected opportunity appeared suddenly.",
        "I never expected such a positive reaction.",
        "The sudden result surprised everyone.",
        "I was amazed by the unexpected achievement.",
        "The unexpected change caught me completely off guard.",
        "I did not anticipate receiving recognition.",
        "The sudden success was surprising.",
        "I was surprised by the team's reaction.",
        "The unexpected news left me amazed.",
        "I never thought I would receive that feedback.",
        "The sudden decision caught everyone by surprise.",
        "I was shocked by the unexpected outcome.",
        "The unexpected result was astonishing.",
        "I did not expect the opportunity to appear.",
        "The sudden change surprised me greatly.",
        "I was amazed by the unexpected support.",
        "The unexpected announcement caught me off guard.",
        "I never anticipated such a result.",
        "The sudden promotion surprised me.",
        "I was shocked when I heard the news.",
        "The unexpected response amazed me.",
        "I did not expect the meeting to change.",
        "The sudden opportunity was a surprise.",
        "I was surprised by the quick progress.",
        "The unexpected success caught me off guard.",
        "I never expected my work to receive recognition.",
        "The sudden announcement was surprising.",
        "I was amazed by the positive development.",
        "The unexpected decision shocked me.",
        "I did not anticipate such a quick solution.",
        "The sudden change was completely unexpected.",
        "I was surprised by my team's reaction.",
        "The unexpected result left me amazed.",
        "I never thought the situation would improve.",
        "The sudden news surprised everyone.",
        "I was shocked by the unexpected opportunity.",
        "The unexpected outcome amazed me.",
        "I did not expect to be chosen.",
        "The sudden success surprised me.",
        "I was amazed by how quickly everything changed.",
        "The unexpected announcement caught me by surprise."
    ],

    "disgust": [
        "I dislike the disrespectful behavior in the workplace.",
        "The toxic environment makes me uncomfortable.",
        "I was disgusted by the way employees were treated.",
        "The rude behavior made me feel sick.",
        "I strongly dislike dishonest workplace practices.",
        "The disrespect shown to my colleague was disgusting.",
        "I feel uncomfortable with the toxic attitude.",
        "The unfair and disrespectful behavior disgusted me.",
        "I was disgusted by the careless treatment of customers.",
        "The negative workplace culture makes me uncomfortable.",
        "I dislike seeing people treated without respect.",
        "The inappropriate behavior was disgusting.",
        "I felt disgusted by the dishonest actions.",
        "The toxic communication made me uncomfortable.",
        "I strongly dislike this kind of disrespect.",
        "The way the issue was handled disgusted me.",
        "I was uncomfortable with the rude comments.",
        "The unethical behavior made me feel disgusted.",
        "I dislike the constant negativity around me.",
        "The disrespectful comments were disgusting.",
        "I felt disgusted by the unfair treatment.",
        "The toxic behavior made me uncomfortable.",
        "I strongly dislike dishonest communication.",
        "The rude attitude disgusted me.",
        "I was uncomfortable watching my colleague being humiliated.",
        "The unethical decision made me feel disgusted.",
        "I dislike the negative behavior in the office.",
        "The disrespect was difficult to tolerate.",
        "I felt disgusted by the way people were treated.",
        "The toxic environment is unpleasant.",
        "I strongly dislike manipulative behavior.",
        "The careless treatment of employees disgusted me.",
        "I felt uncomfortable with the offensive comments.",
        "The dishonest behavior made me disgusted.",
        "I dislike the lack of basic respect.",
        "The toxic atmosphere is disturbing.",
        "I was disgusted by the inappropriate conduct.",
        "The disrespect toward employees made me uncomfortable.",
        "I strongly dislike people who behave dishonestly.",
        "The unethical practices disgusted me.",
        "I felt uncomfortable with the negative culture.",
        "The rude treatment made me feel disgusted.",
        "I dislike seeing coworkers treated badly.",
        "The disrespectful behavior was unpleasant.",
        "I was disgusted by the dishonest explanation.",
        "The toxic attitude made me uncomfortable.",
        "I strongly dislike unfair and unethical behavior.",
        "The offensive remarks disgusted me.",
        "I felt uncomfortable after seeing the disrespect.",
        "The negative behavior was disgusting.",
        "I dislike the way some employees are treated.",
        "The unethical conduct made me feel sick.",
        "I was disgusted by the rude response.",
        "The toxic workplace culture is unpleasant.",
        "I strongly dislike this disrespectful environment.",
        "The careless behavior disgusted me.",
        "I felt uncomfortable with the inappropriate discussion.",
        "The dishonest actions made me disgusted.",
        "I dislike the constant rude behavior.",
        "The disrespect shown during the meeting was disgusting.",
        "I felt disgusted by the toxic attitude.",
        "The unfair treatment was unpleasant to witness.",
        "I strongly dislike unethical decisions.",
        "The offensive behavior made me uncomfortable.",
        "I was disgusted by the lack of respect.",
        "The negative workplace atmosphere is unpleasant.",
        "I dislike seeing dishonest behavior rewarded.",
        "The rude comments disgusted me.",
        "I felt uncomfortable with the toxic conversation.",
        "The disrespectful treatment made me disgusted.",
        "I strongly dislike manipulative workplace behavior.",
        "The unethical actions were disgusting.",
        "I was uncomfortable with the way the situation was handled.",
        "The toxic behavior made me feel sick.",
        "I dislike the disrespect toward junior employees.",
        "The rude treatment disgusted me.",
        "I felt uncomfortable seeing unfair behavior.",
        "The dishonest practice made me disgusted.",
        "I strongly dislike the negative workplace culture.",
        "The offensive comments were unpleasant.",
        "I was disgusted by the disrespectful response.",
        "The toxic environment is difficult to tolerate.",
        "I dislike the careless way people are treated.",
        "The unethical behavior disgusted me.",
        "I felt uncomfortable with the rude attitude.",
        "The disrespectful workplace culture is unpleasant.",
        "I strongly dislike dishonest practices.",
        "The offensive behavior made me disgusted.",
        "I was uncomfortable with the toxic environment.",
        "The unfair conduct was disgusting.",
        "I dislike seeing people treated disrespectfully.",
        "The rude behavior made me feel uncomfortable.",
        "I was disgusted by the negative culture.",
        "The unethical treatment of employees was unpleasant.",
        "I strongly dislike this kind of workplace behavior.",
        "The toxic conduct disgusted me."
    ]
}

def load_training_data():
    """Load and validate the original emotion training dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Input file must contain 'text' and 'label' columns."
        )

    df = df[["text", "label"]].copy()

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df[df["text"] != ""]

    return df


def build_expanded_dataset(original_df):
    """Build a balanced dataset with the target size per emotion."""

    final_rows = []

    for emotion in EMOTIONS:
        existing = original_df[
            original_df["label"] == emotion
        ]["text"].tolist()

        additional = additional_data.get(
            emotion,
            [],
        )

        # Remove duplicates while preserving order.
        combined = list(
            dict.fromkeys(
                existing + additional
            )
        )

        if len(combined) < TARGET_PER_EMOTION:
            raise ValueError(
                f"Not enough unique examples for {emotion}. "
                f"Found {len(combined)}, "
                f"need {TARGET_PER_EMOTION}."
            )

        # Keep every original example.
        selected = existing.copy()

        remaining = [
            text
            for text in combined
            if text not in existing
        ]

        random.shuffle(remaining)

        needed = (
            TARGET_PER_EMOTION
            - len(selected)
        )

        selected.extend(
            remaining[:needed]
        )

        for text in selected:
            final_rows.append(
                {
                    "text": text,
                    "label": emotion,
                }
            )

    return pd.DataFrame(final_rows)


def validate_dataset(df):
    """Validate uniqueness and class balance."""

    if df.empty:
        raise ValueError(
            "Expanded dataset is empty."
        )

    duplicate_count = (
        df["text"].duplicated().sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} "
            "duplicate texts."
        )

    counts = df["label"].value_counts()

    for emotion in EMOTIONS:
        count = counts.get(
            emotion,
            0,
        )

        if count != TARGET_PER_EMOTION:
            raise ValueError(
                f"{emotion} has {count} samples. "
                f"Expected {TARGET_PER_EMOTION}."
            )


def main():
    """Create, validate, shuffle, and save the expanded dataset."""

    random.seed(RANDOM_SEED)

    original_df = load_training_data()

    final_df = build_expanded_dataset(
        original_df
    )

    validate_dataset(
        final_df
    )

    # Shuffle the final dataset deterministically.
    final_df = (
        final_df
        .sample(
            frac=1,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print("EXPANDED EMOTION DATASET CREATED")
    print("=" * 70)

    print(
        f"Total samples: {len(final_df)}"
    )

    print("\nLabel distribution:")
    print(
        final_df["label"].value_counts()
    )

    print("\nDuplicate texts:")
    print(
        final_df["text"].duplicated().sum()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
