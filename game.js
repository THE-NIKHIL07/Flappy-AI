const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

const startScreen = document.getElementById("start-screen");
const livesScreen = document.getElementById("lives-screen");
const endScreen = document.getElementById("end-screen");
const bestScoreText = document.getElementById("best-score");
const modeToggle = document.getElementById("mode-toggle");

const birdImg = new Image();
birdImg.src = "/assets/bird.png";

const bgImg = new Image();
bgImg.src = "/assets/game_bg.png";

const flapSound = new Audio("/assets/flap.mp3");
const scoreSound = new Audio("/assets/score.mp3");
const dieSound = new Audio("/assets/die.mp3");

let audioUnlocked = false;

function unlockAudio() {
    if (audioUnlocked) return;
    flapSound.play().then(() => {
        flapSound.pause();
        flapSound.currentTime = 0;
        audioUnlocked = true;
    }).catch(() => {});
}

document.addEventListener("click", unlockAudio);
document.addEventListener("keydown", unlockAudio);
document.addEventListener("touchstart", unlockAudio);

let width, height;
let birdY, birdVel;
let pipeX, pipeY, lastPipeY;

let score = 0;
let bestScore = 0;
let lives = 3;
let passedPipe = false;
let running = false;

const collisionW = 34;
const collisionH = 24;

const GAP = 130;
const pipeSpeed = 4;

const AI_GRAVITY = 0.8;
const AI_FLAP = -7;

const HUMAN_GRAVITY = 0.45;
const HUMAN_FLAP = -8;

let isAI = true;

let session = null;
let modelLoaded = false;

const FPS = 60;
const frameTime = 1000 / FPS;
let lastTime = 0;

function resize(){
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
}
window.addEventListener("resize", resize);
resize();

function random(min,max){
    return Math.floor(Math.random()*(max-min+1))+min;
}

function nextPipeHeight(){
    let change;
    if(Math.random() < 0.4){
        change = random(-170,170);
    } else {
        change = random(-90,90);
    }
    let newY = lastPipeY + change;
    newY = Math.max(80, Math.min(420,newY));
    lastPipeY = newY;
    return newY;
}

function resetGame(){
    birdY = 256;
    birdVel = 0;
    pipeX = 450;
    pipeY = random(200,300);
    lastPipeY = pipeY;
    score = 0;
    passedPipe = false;
}

async function loadModel(){
    try{
        session = await ort.InferenceSession.create("/models/dueling_dqn_model.onnx");
        modelLoaded = true;
    }catch(e){
        console.error(e);
    }
}

async function aiDecision(){
    if(!modelLoaded || !running) return;
    try{
        const state = new Float32Array([
            birdY / height,
            birdVel / 10,
            pipeX / width,
            pipeY / height,
            (pipeY - birdY) / height,
            (pipeY + GAP - birdY) / height,
            (pipeY + GAP/2 - birdY) / height
        ]);

        const tensor = new ort.Tensor("float32", state, [1,7]);
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;

        const results = await session.run(feeds);
        const output = results[session.outputNames[0]].data;

        if(output[1] > output[0]){
            flap();
        }

    }catch(e){
        console.error(e);
    }
}

function flap(){
    if(isAI){
        birdVel = AI_FLAP;
    }else{
        birdVel = HUMAN_FLAP;
    }
    if(audioUnlocked){
        flapSound.currentTime = 0;
        flapSound.play().catch(()=>{});
    }
}

function handleDeath(){
    if(audioUnlocked){
        dieSound.currentTime = 0;
        dieSound.play().catch(()=>{});
    }
    lives--;
    if(lives <= 0){
        running = false;
        canvas.classList.add("hidden");
        endScreen.classList.remove("hidden");
        bestScoreText.innerText = "Best Score: " + bestScore;
    }else{
        resetGame();
    }
}

function update(){

    if(isAI){
        aiDecision();
        birdVel += AI_GRAVITY;
    }else{
        birdVel += HUMAN_GRAVITY;
    }

    birdY += birdVel;
    pipeX -= pipeSpeed;

    if(!passedPipe && pipeX + 52 < 50){
        score++;
        if(audioUnlocked){
            scoreSound.currentTime = 0;
            scoreSound.play().catch(()=>{});
        }
        passedPipe = true;
        if(score > bestScore) bestScore = score;
    }

    if(pipeX < -52){
        pipeX = 400;
        pipeY = nextPipeHeight();
        passedPipe = false;
    }

    if(
        birdY < 0 ||
        birdY + collisionH > height ||
        (
            50 + collisionW > pipeX &&
            50 < pipeX + 52 &&
            (birdY < pipeY || birdY + collisionH > pipeY + GAP)
        )
    ){
        handleDeath();
    }
}

function draw(){
    ctx.drawImage(bgImg,0,0,width,height);
    ctx.fillStyle = "green";
    ctx.fillRect(pipeX,0,52,pipeY);
    ctx.fillRect(pipeX,pipeY+GAP,52,height);

    let angle = -birdVel * 3 * Math.PI/180;
    ctx.save();
    ctx.translate(50 + collisionW/2, birdY + collisionH/2);
    ctx.rotate(angle);
    ctx.drawImage(birdImg,-20,-20,40,40);
    ctx.restore();

    ctx.fillStyle = "white";
    ctx.font = "28px Arial";
    ctx.fillText("Score: " + score, 20, 40);
    ctx.fillText("Lives: " + lives, 20, 75);
}

function loop(timestamp){
    if(!running) return;
    if(timestamp - lastTime >= frameTime){
        update();
        draw();
        lastTime = timestamp;
    }
    requestAnimationFrame(loop);
}

modeToggle.onclick = ()=>{
    isAI = !isAI;
    modeToggle.innerText = "Mode: " + (isAI ? "AI" : "Human");
};

document.addEventListener("keydown", e=>{
    if(!isAI && e.code === "Space") flap();
});

canvas.addEventListener("click", ()=>{
    if(!isAI) flap();
});

canvas.addEventListener("touchstart", ()=>{
    if(!isAI) flap();
});

document.getElementById("start-btn").onclick = ()=>{
    startScreen.classList.add("hidden");
    livesScreen.classList.remove("hidden");
};

document.getElementById("enter-lives").onclick = async ()=>{
    let val = parseInt(document.getElementById("lives-input").value);
    if(val > 0){
        lives = val;
        livesScreen.classList.add("hidden");
        canvas.classList.remove("hidden");
        if(isAI){
            await loadModel();
        }
        running = true;
        resetGame();
        requestAnimationFrame(loop);
    }
};

document.getElementById("restart-btn").onclick = ()=>{
    endScreen.classList.add("hidden");
    livesScreen.classList.remove("hidden");
};

document.getElementById("exit-btn").onclick = ()=>{
    endScreen.classList.add("hidden");
    startScreen.classList.remove("hidden");
};
