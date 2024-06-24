import express, { Request, Response, NextFunction } from "express"
import { GameState, InfoResponse, MoveResponse } from './types';
import { getIsTrainingInProgress } from './shared';

export interface BattlesnakeHandlers {
  info: () => Promise<InfoResponse> | InfoResponse;
  start: (gameState: GameState) => Promise<void> | void;
  move: (gameState: GameState) => Promise<MoveResponse>;
  end: (gameState: GameState) => Promise<void> | void;
}

export default function runServer(handlers: BattlesnakeHandlers) {
  const app = express();
  app.use(express.json());

  app.get("/", async (req: Request, res: Response) => {
    if (getIsTrainingInProgress()) {
      res.sendStatus(403);
      return;
    }
    res.send(await handlers.info());
  });

  app.post("/start", async (req: Request, res: Response) => {
    await handlers.start(req.body);
    res.send("ok");
  });

  app.post("/move", async (req: Request, res: Response) => {
    res.send(await handlers.move(req.body));
  });

  app.post("/end", async (req: Request, res: Response) => {
    await handlers.end(req.body);
    res.send("ok");
  });

  app.use(function (req: Request, res: Response, next: NextFunction) {
    res.set("Server", "battlesnake/github/starter-snake-typescript");
    next();
  });

  const host = '0.0.0.0';
  const port = parseInt(process.env.PORT || '8000');

  app.listen(port, host, () => {
    console.log(`Running Battlesnake at http://${host}:${port}...`);
  });
}
