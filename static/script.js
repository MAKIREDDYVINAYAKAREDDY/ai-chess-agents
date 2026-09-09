const PIECES = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚"
};

let state = null;
let selectedSquare = null;
let aiThinking = false;

const boardElement = document.getElementById("board");
const statusElement = document.getElementById("gameStatus");
const turnIndicator = document.getElementById("turnIndicator");
const fenElement = document.getElementById("fen");
const moveHistoryElement = document.getElementById("moveHistory");
const capturedElement = document.getElementById("captured");

const modeElement = document.getElementById("mode");
const humanColorElement = document.getElementById("humanColor");
const stockfishLevelElement = document.getElementById("stockfishLevel");
const levelValueElement = document.getElementById("levelValue");


if (stockfishLevelElement && levelValueElement) {
    levelValueElement.textContent =
        stockfishLevelElement.value;

    stockfishLevelElement.addEventListener(
        "input",
        () => {
            levelValueElement.textContent =
                stockfishLevelElement.value;
        }
    );
}


function squareName(row, col) {
    return String.fromCharCode(97 + col) + (8 - row);
}


function parseFen(fen) {

    const placement =
        fen.split(" ")[0];

    const rows =
        placement.split("/");

    const board = [];

    for (const row of rows) {

        const cells = [];

        for (const char of row) {

            if (/[1-8]/.test(char)) {

                for (
                    let i = 0;
                    i < Number(char);
                    i++
                ) {
                    cells.push(null);
                }

            } else {
                cells.push(char);
            }
        }

        board.push(cells);
    }

    return board;
}


function isCaptureMove(uci) {

    if (!state) return false;

    const target =
        uci.slice(2, 4);

    const board =
        parseFen(state.fen);

    const col =
        target.charCodeAt(0) - 97;

    const row =
        8 - Number(target[1]);

    return board[row][col] !== null;
}


function renderBoard() {

    if (!state || !state.fen) {
        return;
    }

    const board =
        parseFen(state.fen);

    const legalMoves =
        state.legal_moves || [];

    boardElement.innerHTML = "";

    for (let row = 0; row < 8; row++) {

        for (let col = 0; col < 8; col++) {

            const square =
                squareName(row, col);

            const piece =
                board[row][col];

            const element =
                document.createElement("div");

            element.className =
                "square " +
                (
                    (row + col) % 2 === 0
                        ? "light"
                        : "dark"
                );

            element.dataset.square =
                square;

            if (selectedSquare === square) {
                element.classList.add(
                    "selected"
                );
            }

            if (selectedSquare) {

                const move =
                    selectedSquare + square;

                const matchingMoves =
                    legalMoves.filter(
                        candidate =>
                            candidate.startsWith(move)
                    );

                if (matchingMoves.length > 0) {

                    element.classList.add(
                        "legal"
                    );

                    if (
                        isCaptureMove(
                            matchingMoves[0]
                        )
                    ) {
                        element.classList.add(
                            "capture"
                        );
                    }
                }
            }

            if (piece) {

                const pieceElement =
                    document.createElement("span");

                pieceElement.className =
                    "piece";

                pieceElement.textContent =
                    PIECES[piece] || piece;

                element.appendChild(
                    pieceElement
                );
            }

            element.addEventListener(
                "click",
                () => {
                    handleSquareClick(square);
                }
            );

            boardElement.appendChild(
                element
            );
        }
    }
}


function render() {

    if (!state) return;

    renderBoard();

    if (turnIndicator) {

        turnIndicator.textContent =
            state.turn === "white"
                ? "White"
                : "Black";
    }

    if (statusElement) {

        if (aiThinking) {

            statusElement.textContent =
                "AI thinking...";

        } else if (state.is_checkmate) {

            statusElement.textContent =
                "Checkmate";

        } else if (state.is_stalemate) {

            statusElement.textContent =
                "Stalemate";

        } else if (state.game_over) {

            statusElement.textContent =
                "Game Over";

        } else if (state.is_check) {

            statusElement.textContent =
                "Check";

        } else {

            statusElement.textContent =
                "Ready";
        }
    }

    if (fenElement) {
        fenElement.textContent =
            state.fen;
    }

    renderMoveHistory();
}


function renderMoveHistory() {

    if (!moveHistoryElement) {
        return;
    }

    const moves =
        state.moves || [];

    if (moves.length === 0) {

        moveHistoryElement.innerHTML =
            "No moves yet.";

        return;
    }

    moveHistoryElement.innerHTML = "";

    for (
        let i = 0;
        i < moves.length;
        i++
    ) {

        const move = moves[i];

        const row =
            document.createElement("div");

        row.className =
            "move-row";

        row.textContent =
            `${i + 1}. ${move.san || move.uci}`;

        moveHistoryElement.appendChild(
            row
        );
    }
}


function showError(message) {

    console.error(
        "Chess UI:",
        message
    );

    if (statusElement) {
        statusElement.textContent =
            message;
    }
}


function isHumanTurn() {

    if (!state) {
        return false;
    }

    if (
        state.mode ===
        "human_vs_stockfish"
    ) {
        return state.turn ===
            state.human_color;
    }

    if (
        state.mode ===
        "human_vs_ppo"
    ) {
        return state.turn ===
            state.human_color;
    }

    return false;
}


function handleSquareClick(square) {

    if (!state) return;

    if (aiThinking) return;

    if (state.game_over) return;

    /*
     * PPO vs Stockfish has no human moves.
     */
    if (!isHumanTurn()) {
        return;
    }

    const board =
        parseFen(state.fen);

    const row =
        8 - Number(square[1]);

    const col =
        square.charCodeAt(0) - 97;

    const piece =
        board[row][col];

    /*
     * Select piece.
     */
    if (!selectedSquare) {

        if (!piece) {
            return;
        }

        const isWhitePiece =
            piece === piece.toUpperCase();

        const pieceColor =
            isWhitePiece
                ? "white"
                : "black";

        if (
            pieceColor !==
            state.turn
        ) {
            return;
        }

        selectedSquare =
            square;

        renderBoard();

        return;
    }

    /*
     * Deselect.
     */
    if (
        selectedSquare === square
    ) {

        selectedSquare = null;

        renderBoard();

        return;
    }

    let uci =
        selectedSquare + square;

    const matchingMoves =
        (state.legal_moves || [])
            .filter(
                move =>
                    move.startsWith(uci)
            );

    if (
        matchingMoves.length === 0
    ) {

        selectedSquare = null;

        renderBoard();

        return;
    }

    /*
     * Promotion.
     */
    if (
        matchingMoves.length > 1
    ) {

        const queenPromotion =
            matchingMoves.find(
                move =>
                    move.endsWith("q")
            );

        uci =
            queenPromotion ||
            matchingMoves[0];

    } else {

        uci =
            matchingMoves[0];
    }

    selectedSquare = null;

    sendHumanMove(uci);
}


async function sendHumanMove(uci) {

    try {

        const response =
            await fetch(
                "/api/move",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        uci: uci
                    })
                }
            );

        const data =
            await response.json();

        if (!data.success) {

            showError(
                data.error ||
                "Move failed"
            );

            return;
        }

        state =
            data.state;

        render();

        /*
         * Human vs Stockfish
         */
        if (
            state.mode ===
            "human_vs_stockfish" &&
            state.turn !==
            state.human_color &&
            !state.game_over
        ) {

            await makeStockfishMove();
        }

        /*
         * Human vs PPO
         */
        else if (
            state.mode ===
            "human_vs_ppo" &&
            state.turn !==
            state.human_color &&
            !state.game_over
        ) {

            await makePPOMove();
        }

    } catch (error) {

        showError(
            error.message
        );
    }
}


async function makeStockfishMove() {

    if (aiThinking) return;

    aiThinking = true;

    render();

    try {

        const response =
            await fetch(
                "/api/ai-move",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        stockfish_level:
                            Number(
                                stockfishLevelElement.value
                            )
                    })
                }
            );

        const data =
            await response.json();

        if (!data.success) {

            showError(
                data.error ||
                "Stockfish move failed"
            );

            return;
        }

        state =
            data.state;

    } catch (error) {

        showError(
            error.message
        );

    } finally {

        aiThinking = false;

        render();
    }
}


async function makePPOMove() {

    if (aiThinking) return;

    aiThinking = true;

    render();

    try {

        const response =
            await fetch(
                "/api/ppo-move",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    }
                }
            );

        const data =
            await response.json();

        if (!data.success) {

            showError(
                data.error ||
                "PPO move failed"
            );

            return;
        }

        state =
            data.state;

    } catch (error) {

        showError(
            error.message
        );

    } finally {

        aiThinking = false;

        render();
    }
}


async function makeAIVsAIMove() {
    if (aiThinking) return;

    if (
        !state ||
        state.game_over ||
        state.mode !== "ppo_vs_stockfish"
    ) {
        return;
    }

    aiThinking = true;
    render();

    try {
        const response = await fetch("/api/ai-vs-ai", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            showError(data.error || "AI move failed");
            return;
        }

        state = data.state;

    } catch (error) {
        console.error("AI vs AI error:", error);
        showError(error.message);

    } finally {
        aiThinking = false;
        render();

        // Continue automatically while the game is running.
        if (
            state &&
            state.mode === "ppo_vs_stockfish" &&
            !state.game_over
        ) {
            window.aiLoopTimer = setTimeout(
                makeAIVsAIMove,
                350
            );
        }
    }
}

async function newGame() {
    // Prevent duplicate reset requests
    if (window.resettingGame) {
        return;
    }

    window.resettingGame = true;

    try {
        const modeElement = document.getElementById("mode");
        const colorElement = document.getElementById("humanColor");
        const levelElement = document.getElementById("stockfishLevel");

        const mode = modeElement ? modeElement.value : "human_vs_stockfish";
        const human_color = colorElement ? colorElement.value : "white";
        const stockfish_level = levelElement
            ? parseInt(levelElement.value || "5", 10)
            : 5;

        const response = await fetch("/api/new", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                mode: mode,
                human_color: human_color,
                stockfish_level: stockfish_level
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not reset game");
        }

        // Stop any previous AI loop
        if (window.aiLoopTimer) {
            clearTimeout(window.aiLoopTimer);
            window.aiLoopTimer = null;
        }

        state = data.state;
        render();

        // Let AI make the first move when human is Black
        if (
            state.mode === "human_vs_stockfish" &&
            state.human_color === "black"
        ) {
            await makeStockfishMove();
        } else if (
            state.mode === "human_vs_ppo" &&
            state.human_color === "black"
        ) {
            await makePPOMove();
        } else if (state.mode === "ppo_vs_stockfish") {
            await makeAIVsAIMove();
        }

    } catch (error) {
        console.error("New game error:", error);
        alert("Could not reset game: " + error.message);
    } finally {
        window.resettingGame = false;
    }
}

async function undo() {

    if (aiThinking) return;

    selectedSquare = null;

    try {

        const response =
            await fetch(
                "/api/undo",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!data.success) {

            showError(
                data.error ||
                "Nothing to undo"
            );

            return;
        }

        state =
            data.state;

        render();

    } catch (error) {

        showError(
            error.message
        );
    }
}


document.addEventListener(
    "DOMContentLoaded",
    async () => {

        const newGameButton =
            document.getElementById(
                "newGame"
            );

        const undoButton =
            document.getElementById(
                "undo"
            );

        if (newGameButton) {

            newGameButton.addEventListener(
                "click",
                newGame
            );
        }

        if (undoButton) {

            undoButton.addEventListener(
                "click",
                undo
            );
        }

        /*
         * Load current state.
         */
        try {

            const response =
                await fetch(
                    "/api/state"
                );

            const data =
                await response.json();

            if (data.success) {

                state =
                    data.state;

                render();
            }

        } catch (error) {

            console.error(
                "Initial state error:",
                error
            );
        }
    }
);

/* Reset button */
document.addEventListener("DOMContentLoaded", () => {
    const resetButton = document.getElementById("reset");

    if (resetButton) {
        resetButton.addEventListener("click", () => {
            newGame();
        });
    }
});
