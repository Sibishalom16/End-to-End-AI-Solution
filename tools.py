"""Custom deterministic study tools for the Study Planner AI Agent.

Provides calculation engines for:
1. calculate_study_hours: Computes study budgets and weekly capacity with validation.
2. generate_study_plan: Synthesizes structured study schedules with revision and breaks.
"""

from typing import Union, List
from langchain_core.tools import tool


def calculate_study_hours(
    available_days: Union[int, float, str],
    hours_per_day: Union[int, float, str]
) -> str:
    """Calculate total available study hours given available days and daily study hours.

    Args:
        available_days: Total number of days available before the deadline or exam.
        hours_per_day: Estimated productive study hours per day.

    Returns:
        A structured string containing total study budget, weekly pacing,
        and Pomodoro session estimates, or a descriptive validation error message.
    """
    # 1. Type validation and coercion
    try:
        days = float(available_days)
    except (ValueError, TypeError):
        return f"Validation Error: 'available_days' must be a valid positive number. Received: {available_days}"

    try:
        daily_hours = float(hours_per_day)
    except (ValueError, TypeError):
        return f"Validation Error: 'hours_per_day' must be a valid positive number. Received: {hours_per_day}"

    # 2. Value boundary validation
    if days <= 0:
        return f"Validation Error: 'available_days' must be greater than 0. Received: {days}."
    
    if days > 365:
        return f"Validation Error: 'available_days' cannot exceed 365 days (1 year) for a study plan. Received: {days}."

    if daily_hours <= 0:
        return f"Validation Error: 'hours_per_day' must be greater than 0. Received: {daily_hours}."

    if daily_hours > 16:
        return (
            f"Validation Error: 'hours_per_day' cannot exceed 16 hours (received {daily_hours}). "
            "Studying more than 16 hours per day is physically unsustainable and leads to severe cognitive burnout."
        )

    # 3. Calculations
    total_hours = round(days * daily_hours, 2)
    # Each standard Pomodoro block is 25 min work + 5 min break = 0.5 hour total
    estimated_pomodoros = int(total_hours / 0.5)
    
    # Weekly equivalent
    weeks = days / 7.0
    hours_per_week = round(daily_hours * 7.0, 2)

    # Workload intensity classification
    if daily_hours <= 2:
        intensity = "Light / Maintenance (Ideal for long-term consistency)"
    elif daily_hours <= 5:
        intensity = "Moderate / High Yield (Optimal for exam revision)"
    elif daily_hours <= 8:
        intensity = "Intensive / Full-Time Study (Requires strict rest breaks)"
    else:
        intensity = "Extreme Sprint (Only sustainable for 2-3 consecutive days)"

    result = [
        "### 📊 Study Hours Calculation Report",
        f"- **Available Time**: {int(days) if days.is_integer() else days} day(s)",
        f"- **Daily Study Target**: {daily_hours:.1f} hour(s) per day",
        f"- **Total Study Budget**: **{total_hours:.1f} hour(s)**",
        f"- **Weekly Equivalent**: ~{hours_per_week:.1f} hours/week across {weeks:.1f} week(s)",
        f"- **Estimated Pomodoro Blocks**: ~{estimated_pomodoros} sessions (25 min study / 5 min break)",
        f"- **Pacing Assessment**: {intensity}",
        "",
        "💡 *Recommendation*: Maintain at least 7-8 hours of sleep nightly. Cognitive consolidation during sleep is essential for memory retention."
    ]
    return "\n".join(result)


def generate_study_plan(
    topics: Union[str, List[str]],
    available_days: Union[int, float, str],
    hours_per_day: Union[int, float, str]
) -> str:
    """Generate a structured, day-by-day study schedule distributed across topics.

    Args:
        topics: Comma-separated string or list of topic names to cover.
        available_days: Number of days available until the deadline/exam.
        hours_per_day: Scheduled study hours per day.

    Returns:
        A formatted Markdown study plan detailing daily topic focus, revision sessions,
        practice testing, and break recommendations, or a validation error.
    """
    # 1. Parse and validate topics
    if isinstance(topics, list):
        topic_list = [str(t).strip() for t in topics if str(t).strip()]
    elif isinstance(topics, str):
        # Split on commas or newlines
        raw_items = topics.replace("\n", ",").split(",")
        topic_list = [item.strip() for item in raw_items if item.strip()]
    else:
        return f"Validation Error: 'topics' must be a comma-separated string or list of topics. Received: {type(topics)}"

    if not topic_list:
        return "Validation Error: 'topics' list cannot be empty. Please provide at least one topic."

    # 2. Validate days and hours
    try:
        days = int(float(available_days))
    except (ValueError, TypeError):
        return f"Validation Error: 'available_days' must be an integer. Received: {available_days}"

    try:
        daily_hours = float(hours_per_day)
    except (ValueError, TypeError):
        return f"Validation Error: 'hours_per_day' must be a valid number. Received: {hours_per_day}"

    if days <= 0:
        return f"Validation Error: 'available_days' must be at least 1 day. Received: {days}."

    if days > 90:
        return (
            f"Validation Error: 'available_days' cannot exceed 90 days for this detailed planner (received {days}). "
            "Please break down plans longer than 3 months into quarterly milestones."
        )

    if daily_hours <= 0:
        return f"Validation Error: 'hours_per_day' must be greater than 0. Received: {daily_hours}."

    if daily_hours > 16:
        return (
            f"Validation Error: 'hours_per_day' exceeds safe limit of 16 hours (received {daily_hours}). "
            "Sustainable academic success requires balanced rest."
        )

    # 3. Schedule Synthesis
    total_study_hours = round(days * daily_hours, 1)
    num_topics = len(topic_list)

    output = [
        "### 📅 Personalized Strategic Study Plan",
        f"- **Plan Horizon**: {days} Day(s) | **Daily Commitment**: {daily_hours:.1f} Hours/Day",
        f"- **Total Study Budget**: **{total_study_hours} Hours**",
        f"- **Topics Covered ({num_topics})**: {', '.join(topic_list)}",
        "",
        "---",
        "#### 🗓️ Day-by-Day Schedule & Milestones:"
    ]

    if days == 1:
        # Single day cram/intensive review schedule
        hours_per_topic = round((daily_hours * 0.75) / num_topics, 2)
        review_hours = round(daily_hours * 0.25, 2)
        output.append("**Day 1: Intensive Single-Day Protocol**")
        for i, topic in enumerate(topic_list, 1):
            output.append(f"  - **Session {i}** ({hours_per_topic:.1f} hrs): Deep dive into **{topic}** (Active Recall & Core Problems)")
        output.append(f"  - **Final Consolidation & Error Review** ({review_hours:.1f} hrs): Synthesis sheets and mock questions")
        output.append("  - *Protocol*: Use 50-minute work blocks with 10-minute rest breaks.")
    else:
        # Multi-day schedule
        # Allocate revision days: For >= 5 days, reserve last 1-2 days for comprehensive revision & mock exams
        if days >= 7:
            revision_days = 2
        elif days >= 4:
            revision_days = 1
        else:
            revision_days = 0

        learning_days = days - revision_days

        # Distribute topics across learning days
        for day in range(1, learning_days + 1):
            # Select topic(s) for the day using round-robin or chunking
            day_topic_idx = (day - 1) % num_topics
            primary_topic = topic_list[day_topic_idx]
            
            # Secondary topic for interleaving if multiple topics exist
            if num_topics > 1:
                secondary_idx = day % num_topics
                secondary_topic = topic_list[secondary_idx]
                topic_focus = f"Primary: **{primary_topic}** (~70% time) | Interleaved Review: **{secondary_topic}** (~30% time)"
            else:
                topic_focus = f"Focus: **{primary_topic}** (Syllabus Audit & Problem Sets)"

            output.append(f"**Day {day}** ({daily_hours:.1f} hrs): {topic_focus}")
            output.append("  - *Activity*: Concept mapping, closed-book flashcards, targeted problem sets.")

        # Revision days
        if revision_days > 0:
            for rev_day in range(learning_days + 1, days + 1):
                if rev_day == days:
                    output.append(
                        f"**Day {rev_day} (Final Triage & Consolidation - {daily_hours:.1f} hrs)**: "
                        "Light active recall, error log review, formula cards, and mandatory early rest."
                    )
                else:
                    output.append(
                        f"**Day {rev_day} (Simulated Exam & Synthesis - {daily_hours:.1f} hrs)**: "
                        "Timed mock exam under strict conditions across all topics, followed by error log diagnosis."
                    )

    # 4. Break & Recovery Recommendations
    if daily_hours <= 3:
        pomo_recommendation = "25 minutes study + 5 minutes break (Standard Pomodoro)"
    else:
        pomo_recommendation = "50 minutes deep study + 10 minutes restorative break (Deep Work Protocol)"

    output.extend([
        "",
        "---",
        "#### ⏱️ Cognitive Architecture & Break Strategy:",
        f"- **Work / Rest Cadence**: {pomo_recommendation}",
        "- **After Every 4 Blocks**: Take a longer 20-30 minute walk or stretching break.",
        "- **Active Recall Rule**: Spend at least 40% of each study block actively testing yourself rather than passively re-reading.",
        "- **Sleep Priority**: Maintain consistent bedtime; sleep flattens the forgetting curve and consolidates memory."
    ])

    return "\n".join(output)


# LangChain Tool wrappers for agent invocation
@tool
def calculate_study_hours_tool(available_days: int, hours_per_day: float) -> str:
    """Calculate total available study hours given available days and daily study hours.
    Use this tool whenever the user asks to calculate study hours, compute total study time,
    or check available study hours for an exam or deadline.
    """
    return calculate_study_hours(available_days=available_days, hours_per_day=hours_per_day)


@tool
def generate_study_plan_tool(topics: str, available_days: int, hours_per_day: float) -> str:
    """Generate a structured, personalized daily study schedule for a set of topics.
    Use this tool whenever the user asks to create, build, or generate a study plan,
    revision timetable, or study schedule for topics over a given number of days.
    """
    return generate_study_plan(topics=topics, available_days=available_days, hours_per_day=hours_per_day)
