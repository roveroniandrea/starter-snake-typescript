// Welcome to
// __________         __    __  .__                               __
// \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
//  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
//  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
//  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
//
// This file can be a nice home for your Battlesnake logic and helper functions.
//
// To get you started we've included code to prevent your Battlesnake from moving backwards.
// For more info see docs.battlesnake.com

import runServer from './server';
import { SnakeAgent } from './snakeAgent';
import { Moves } from './utils';
import { GameState, InfoResponse, MoveResponse } from './types';

const snakeAgent = new SnakeAgent();

let timeStats: {
  gameIndex: number;
  trainingTimeMs: number;
  playingTimeMs: number;
} = {
  gameIndex: 0,
  trainingTimeMs: 0,
  playingTimeMs: 0
}


// info is called when you create your Battlesnake on play.battlesnake.com
// and controls your Battlesnake's appearance
// TIP: If you open your Battlesnake URL in a browser you should see this data
function info(): InfoResponse {
  console.log("INFO");

  return {
    apiversion: "1",
    author: "",       // TODO: Your Battlesnake Username
    color: "#888888", // TODO: Choose color
    head: "default",  // TODO: Choose head
    tail: "default",  // TODO: Choose tail
  };
}

// start is called when your Battlesnake begins a game
function start(gameState: GameState): void {
  // console.log("GAME START");
}

// end is called when your Battlesnake finishes a game
async function end(gameState: GameState): Promise<void> {
  const now = new Date().getTime();
  await snakeAgent.trainAll();
  const finalTrainingTime = new Date().getTime() - now;

  console.log(`
    GAME #${timeStats.gameIndex} ENDED:
    Turns ${gameState.turn}
    Avg training time during game: ${timeStats.trainingTimeMs / gameState.turn}ms
    Avg playing time: ${timeStats.playingTimeMs / gameState.turn}ms
    Final training time: ${finalTrainingTime}ms
    `);

  timeStats = {
    gameIndex: timeStats.gameIndex++,
    trainingTimeMs: 0,
    playingTimeMs: 0
  };
}

// move is called on every turn and returns your next move
// Valid moves are "up", "down", "left", or "right"
// See https://docs.battlesnake.com/api/example-move for available data
async function move(gameState: GameState): Promise<MoveResponse> {
  // if (gameState.turn && false) {
  //   const now = new Date().getTime();
  //   await snakeAgent.train(gameState);

  //   timeStats.trainingTimeMs += (new Date().getTime()) - now;
  // }

  const now = new Date().getTime();
  const nextMove: Moves = await snakeAgent.play(gameState);
  timeStats.playingTimeMs += (new Date().getTime()) - now;

  return { move: nextMove };
}

runServer({
  info: info,
  start: start,
  move: move,
  end: end
});
