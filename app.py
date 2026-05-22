import gc
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from modules.data_utils import get_dataset
from modules.perceptron import Perceptron
from modules.mlp_numpy import MLPNumPy
from modules.mlp_linear import MLPLinear
from modules.plotting import plot_decision_boundary, plot_training_history
from pathlib import Path

ASSET_DIR = Path(__file__).parent / "assets"

st.set_page_config(
    page_title="Perceptron vs MLP",
    page_icon="🧠",
    layout="wide",
)


st.title("🧠 Perceptron vs Neural Network")
st.subheader("Varför en perceptron klarar linjära problem men misslyckas med icke-linjära problem")


with st.sidebar:
    st.header("Inställningar")

    dataset_name = st.selectbox(
        "Välj dataset",
        ["Linear", "XOR", "Moons", "Circles"],
    )

    model_name = st.selectbox(
        "Välj modell",
        ["Single Perceptron", "MLP with hidden layer", "MLP without activation"],
    )

    n_samples = st.slider("Antal datapunkter", 50, 500, 200, 50)
    noise = st.slider(
        "Noise",
        0.0,
        0.6,
        0.2,
        0.05,
        help="Högre noise gör datapunkterna mer slumpmässiga och svårare att separera.",
    )
    learning_rate = st.slider("Learning rate", 0.001, 1.0, 0.1, 0.001)

    if model_name == "Single Perceptron":
        epochs = st.slider("Epochs", 1, 200, 40, 1)
        hidden_neurons = None
    else:
        epochs = st.slider("Epochs", 100, 2000, 1000, 100)
        hidden_neurons = st.slider("Hidden neurons", 2, 20, 4, 1)

    random_state = st.number_input("Random seed", value=42, step=1)

    train_button = st.button("Träna modellen")


X, y = get_dataset(
    dataset_name,
    n_samples=n_samples,
    noise=noise,
    random_state=random_state,
)


col1, col2 = st.columns([1.2, 1])


with col1:
    st.markdown("### Dataset")

    if dataset_name == "Linear":
        st.info("Detta dataset är ungefär linjärt separerbart. En enkel perceptron borde fungera bra.")
    elif dataset_name == "XOR":
        st.warning("XOR är inte linjärt separerbart. En enkel perceptron brukar misslyckas.")
    else:
        st.warning("Detta dataset kräver en icke-linjär beslutsgräns. En enkel perceptron är oftast för enkel.")

    st.markdown("### Modellstruktur")

    if model_name == "Single Perceptron":
        st.image(
            ASSET_DIR / "perceptron.png",
            caption="Enkel perceptron: linjär kombination + step-funktion",
            width="stretch",
        )
    else:
        st.image(
            ASSET_DIR / "mlp.png",
            caption="MLP: hidden layer med sigmoid-aktivering + output sigmoid",
            width="stretch",
        )    

    if train_button:
        for key in ["model", "history", "X", "y", "epochs", "model_name", "dataset_name"]:
            if key in st.session_state:
                del st.session_state[key]

        gc.collect()

        if model_name == "Single Perceptron":
            model = Perceptron(
                learning_rate=learning_rate,
                random_state=random_state,
            )
            history = model.fit(X, y, epochs=epochs)

        elif model_name == "MLP with hidden layer":
            model = MLPNumPy(
                hidden_neurons=hidden_neurons,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            history = model.fit(X, y, epochs=epochs)

        else:
            model = MLPLinear(
                hidden_neurons=hidden_neurons,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            history = model.fit(X, y, epochs=epochs)

        st.session_state["model"] = model
        st.session_state["history"] = history
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["epochs"] = epochs
        st.session_state["model_name"] = model_name
        st.session_state["dataset_name"] = dataset_name


    if "model" in st.session_state:
        model = st.session_state["model"]
        history = st.session_state["history"]
        X_saved = st.session_state["X"]
        y_saved = st.session_state["y"]
        epochs_saved = st.session_state["epochs"]
        model_name_saved = st.session_state["model_name"]
        dataset_name_saved = st.session_state["dataset_name"]

        st.markdown("### Decision boundary under träning")

        show_epoch_animation = st.checkbox(
            "Visa decision boundary epoch-by-epoch",
            value=True,
        )

        if show_epoch_animation:
            selected_epoch = st.slider(
                "Välj epoch",
                1,
                epochs_saved,
                epochs_saved,
                1,
            )

            if model_name_saved == "Single Perceptron":
                model.set_params(
                    history["weights"][selected_epoch - 1],
                    history["bias"][selected_epoch - 1],
                )
            else:
                model.set_params(
                    history["params"][selected_epoch - 1],
                )
        else:
            selected_epoch = epochs_saved

        st.caption(f"Visar modellens beslutsgräns efter epoch {selected_epoch}.")

        fig_boundary = plot_decision_boundary(
            model,
            X_saved,
            y_saved,
            title=f"{model_name_saved} on {dataset_name_saved} — epoch {selected_epoch}",
        )

        st.pyplot(fig_boundary)
        plt.close(fig_boundary)
        gc.collect()

    else:
        st.write("Tryck på **Träna modellen** för att se beslutsgränsen.")

with col2:
    st.markdown("### Resultat och förklaring")

    if "model" in st.session_state:
        history = st.session_state["history"]
        model_name_saved = st.session_state["model_name"]
        dataset_name_saved = st.session_state["dataset_name"]

        final_accuracy = history["accuracy"][-1]

        st.metric("Final accuracy", f"{final_accuracy:.2%}")

        fig_history = plot_training_history(history, model_name_saved)
        st.pyplot(fig_history)
        plt.close(fig_history)
        gc.collect()

        if model_name_saved == "Single Perceptron":
            st.markdown(
                """
                **Perceptronens begränsning**

                En enkel perceptron använder en linjär kombination:

                ```text
                z = w1*x1 + w2*x2 + b
                ```

                och en step-funktion:

                ```text
                z ≥ 0 → klass 1
                z < 0  → klass 0
                ```

                Beslutsgränsen uppstår där modellen byter klass, alltså där `z = 0`.

                Därför blir gränsen en rak linje.

                Det fungerar bra för linjärt separerbara problem, men inte för XOR,
                moons eller circles.
                """
            )

            if dataset_name_saved != "Linear":
                st.error(
                    "Här ser vi huvudpoängen: problemet är inte bara träningen. "
                    "Modellen är för enkel för att representera en icke-linjär lösning."
                )

        elif model_name_saved == "MLP with hidden layer":
            st.markdown(
                """
                **Varför hjälper hidden layers + sigmoid?**

                Denna modell har ett hidden layer med flera neuroner.

                Varje hidden neuron beräknar först en linjär kombination:

                ```text
                z = w1*x1 + w2*x2 + b
                ```

                Sedan används sigmoid:

                ```text
                h = sigmoid(z)
                ```

                Det betyder att varje hidden neuron skickar vidare ett mjukt värde
                mellan 0 och 1, inte bara klass 0 eller klass 1.

                Output-neuronen kombinerar sedan dessa hidden activations:

                ```text
                z_out = v1*h1 + v2*h2 + ... + b_out
                y_pred = sigmoid(z_out)
                ```

                Till sist används threshold:

                ```text
                y_pred ≥ 0.5 → klass 1
                y_pred < 0.5  → klass 0
                ```

                Eftersom modellen kombinerar flera neuroner med icke-linjära
                aktiveringsfunktioner kan den skapa mer flexibla beslutsgränser.
                """
            )

            if dataset_name_saved != "Linear":
                st.success(
                    "Nu kan modellen ofta hitta en icke-linjär lösning eftersom den har "
                    "hidden layer och sigmoid-aktivering."
                )

        elif model_name_saved == "MLP without activation":
            st.markdown(
                """
                **Varför räcker inte flera lager utan aktiveringsfunktion?**

                Denna modell ser ut som ett neuralt nätverk eftersom den har ett hidden layer.

                Men hidden layer saknar icke-linjär aktiveringsfunktion.

                Då blir modellen i princip:

                ```text
                linear layer → linear layer → sigmoid output
                ```

                Problemet är att flera linjära transformationer efter varandra fortfarande
                kan förenklas till en enda linjär transformation:

                ```text
                Linear + Linear = Linear
                ```

                Därför kan modellen fortfarande bara skapa en linjär beslutsgräns före
                sista sigmoid-funktionen.

                Den kan fungera på linjära dataset, men brukar misslyckas på XOR,
                moons och circles.
                """
            )

            if dataset_name_saved != "Linear":
                st.warning(
                    "Här ser vi en viktig ANN-insikt: hidden layers räcker inte ensamma. "
                    "Vi behöver också icke-linjära aktiveringsfunktioner."
                )

    else:
        st.markdown(
            """
            Testa gärna denna ordning:

            1. **Linear + Single Perceptron**
            2. **XOR + Single Perceptron**
            3. **XOR + MLP without activation**
            4. **XOR + MLP with hidden layer**

            Då blir skillnaden mellan linjär modell, flera linjära lager och riktig
            icke-linjär neural network-modell tydlig.
            """
        )


st.markdown("---")

with st.expander("Pedagogisk sammanfattning"):
    st.markdown(
        """
        # Hur fungerar en perceptron?

        En enkel perceptron tar emot flera inputvariabler och beräknar en viktad summa:

        ```text
        z = w1*x1 + w2*x2 + b
        ```

        där:

        - `x1`, `x2` = inputvariabler
        - `w1`, `w2` = vikter
        - `b` = bias

        Därefter används en aktiveringsfunktion för att bestämma klass:

        ```text
        om z ≥ 0 → klass 1
        om z < 0 → klass 0
        ```

        ## Varför blir beslutsgränsen linjär?

        Själva beslutsgränsen uppstår där modellen är osäker:

        ```text
        z = 0
        ```

        alltså:

        ```text
        w1*x1 + w2*x2 + b = 0
        ```

        Detta är ekvationen för:

        - en rak linje i 2D
        - ett plan i 3D
        - ett hyperplan i högre dimensioner

        Därför kan en enkel perceptron bara skapa linjära beslutsgränser.

        ## Vad betyder XOR?

        XOR betyder "Exclusive OR".

        Output blir 1 endast när exakt en av inputvariablerna är 1.

        XOR-tabell:

        ```text
        x1   x2   y
        0    0    0
        0    1    1
        1    0    1
        1    1    0
        ```

        I 2D-planet hamnar klasserna diagonalt:

        ```text
        Klass 1:
        övre vänster + nedre höger

        Klass 0:
        övre höger + nedre vänster
        ```

        Ingen enda rak linje kan separera dessa grupper perfekt.

        Därför misslyckas en enkel perceptron på XOR-problemet.

        ## Varför fungerar MLP bättre?

        Ett Multi-Layer Perceptron (MLP) innehåller hidden layers med flera neuroner.

        Varje neuron kan skapa en enkel linjär gräns.

        När flera neuroner kombineras kan nätverket bygga mer komplexa,
        icke-linjära beslutsgränser.

        Därför kan MLP lösa problem som:

        - XOR
        - circles
        - moons

        trots att en enkel perceptron inte kan göra det.
        """
    )