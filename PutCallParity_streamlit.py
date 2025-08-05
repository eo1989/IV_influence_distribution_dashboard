import random as rd

import streamlit as st


def get_random_value(_min, _max, decimals=0):
    return round(rd.uniform(_min, _max), decimals)


def display_question(missing_variable=None, correct_value=None):
    if missing_variable is None or correct_value is None:
        options = ["spot", "strike", "carry", "call", "put"]
        missing_variable = rd.choice(options)

        spot = get_random_value(10, 110, 2)
        strike = get_random_value(0.9 * spot, 1.1 * spot, 0)
        carry = get_random_value(-0.05 * spot, 0.05 * spot, 2)

        if spot > strike:
            put = get_random_value(0, 0.1 * strike, 2)
        elif spot < strike:
            put = round(
                get_random_value(0, 0.1 * strike, 2) + (strike - spot), 2
            )

        call = max(0, round(spot - strike + put + carry, 2))

        correct_values = {
            "spot": spot,
            "strike": strike,
            "carry": carry,
            "call": call,
            "put": put,
        }

    st.write(
        f"Questions remaining: {st.session_state.num_games - st.session_state.correct_answers}"
    )

    if missing_variable != "spot":
        st.write(f"Spot: {spot}")
    if missing_variable != "strike":
        st.write(f"Strike: {strike}")
    if missing_variable != "carry":
        st.write(f"carry: {carry}")
    if missing_variable != "call":
        st.write(f"call: {call}")
    if missing_variable != "put":
        st.write(f"put: {put}")

    user_input = st.text_input(f"Enter the value for {missing_variable}")

    if user_input:
        user_value = float(user_input)
        if correct_values[missing_variable] == user_value:
            st.session_state.correct_answers += 1
            st.success("Correct")
            if st.session_state.correct_answers < st.session_state.num_games:
                display_question()
            else:
                check_game_end()

        else:
            st.error("Incorrect, try again.")
            display_question(missing_variable, correct_values)


def check_game_end():
    st.write(
        f"Game over. You reached {st.session_state.correct_answers} correct answers out of {st.session_state.num_games}."
    )
    restart_button = st.button("Play again?")
    if restart_button:
        st.session_state.correct_answers = 0
        st.session_state.num_games = st.number_input(
            "How many correct answers do you want to achieve?",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )
        display_question()


if __name__ == "__main__":
    st.title("Put Call parity trial")

    if "correct_answers" not in st.session_state:
        st.session_state.correct_answers = 0
    if "num_games" not in st.session_state:
        st.session_state.num_games = st.number_input(
            "How many correct answers do you want to achieve?",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )

    if st.session_state.correct_answers < st.session_state.num_games:
        display_question()
    else:
        check_game_end()
