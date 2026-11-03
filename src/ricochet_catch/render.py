"""Local browser display. Playback is only recorded frames; live play uses Env."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
import webbrowser

from .env import Env, GameConfig, make_tape

PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ricochet Catch</title>
<style>
:root{color-scheme:dark;font-family:ui-sans-serif,system-ui,sans-serif;background:#0c101b;color:#ecf1ff}
*{box-sizing:border-box}body{margin:0;padding:38px 26px}main{max-width:1020px;margin:auto}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:26px}h1{font-size:32px;letter-spacing:-1.1px;margin:0}small,.muted{color:#9aabc7}small{letter-spacing:2px;font-size:11px}
.pill{padding:8px 13px;border:1px solid #35405c;border-radius:20px;font-size:12px;color:#a9bbdb}
.layout{display:grid;grid-template-columns:minmax(260px,620px) 260px;gap:26px}.board{background:#131b2c;border:1px solid #35405c;border-radius:16px;overflow:hidden;aspect-ratio:1}svg{display:block;width:100%;height:100%}
aside{display:flex;flex-direction:column;gap:16px}.card{background:#141d2f;border:1px solid #2b3850;border-radius:13px;padding:20px}.big{font-size:48px;line-height:1.2;font-weight:650;letter-spacing:-2px}.label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#9aabc7}.row{display:flex;justify-content:space-between;gap:12px;margin-top:13px;font-size:14px}.status{color:#5eebc4;font-size:14px;margin-top:14px;min-height:20px}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px;align-items:center}button,select{font:inherit;background:#202d47;color:#eef4ff;border:1px solid #3b4c6b;border-radius:8px;padding:10px 15px;cursor:pointer}button:hover{background:#30415f}button.primary{background:#65edcc;color:#09291f;border-color:#65edcc;font-weight:650}input{accent-color:#65edcc}.timeline{width:100%;margin-top:18px}.help{font-size:12px;line-height:1.8;color:#9aabc7;margin:18px 0}kbd{border:1px solid #4a5873;padding:2px 5px;border-radius:4px;color:#d6e1f7}.legend{display:flex;gap:20px;font-size:13px;color:#b6c7df;margin-top:14px}.diamond{color:#5eebc4}.cross{color:#ff8193}.note{font-size:12px;line-height:1.6;color:#aab9d2;overflow-wrap:anywhere}#logits{font-family:ui-monospace,monospace;font-size:12px;margin-top:12px;line-height:1.8}
@media(max-width:760px){body{padding:20px 14px}.layout{grid-template-columns:1fr}aside{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{padding:15px}h1{font-size:26px}header{gap:10px}.big{font-size:38px}}
</style>
<main><header><div><small>RICOCHET / 01</small><h1>Ricochet Catch</h1></div><span class="pill" id="mode"></span></header>
<div class="layout"><section><div class="board"><svg id="board" viewBox="0 0 640 640" role="img" aria-label="Eight by eight game board"></svg></div>
<div class="legend"><span class="diamond">◆ Gem +1</span><span class="cross">✕ Bomb ends play</span><span>▬ Paddle</span></div>
<div class="controls"><button id="pause" class="primary">Play</button><button id="step">Step →</button><button id="restart">Restart</button><select id="speed" aria-label="Ticks per second"><option value="2">2 ticks / sec</option><option value="5" selected>5 ticks / sec</option><option value="10">10 ticks / sec</option><option value="20">20 ticks / sec</option></select></div>
<input class="timeline" id="timeline" type="range" min="0" value="0" aria-label="Replay tick">
<p class="help" id="help"></p></section>
<aside><div class="card"><div class="label">Gems caught</div><div class="big" id="score">0</div><div class="row"><span class="muted">Tick</span><span id="tick"></span></div><div class="row"><span class="muted">Last action</span><span id="action">—</span></div><div class="status" id="status">Ready</div></div>
<div class="card"><div class="label">Policy</div><p id="policy"></p><div class="label">Selection</div><p class="note" id="selection"></p><div id="logits"></div></div></aside></div></main>
<script>
const data=__DATA__, live=__LIVE__, frames=data.trajectory||[];
let index=0,paused=true,inFlight=false,pendingRestart=false,left=false,right=false,timer;
const $=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
function shape(tag,attrs){const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));$('board').appendChild(el);return el}
function draw(){const f=frames[index];if(!f)return;const s=f.state;$('board').replaceChildren();
for(let r=0;r<8;r++)for(let c=0;c<8;c++)shape('rect',{x:c*80,y:r*80,width:80,height:80,fill:r===7?'#1b2942':((r+c)%2?'#152036':'#131c2f'),stroke:'#27334a','stroke-width':.6});
for(const o of s.objects){const x=o.column*80+40,y=o.row*80+34;if(o.kind==='gem')shape('polygon',{points:`${x},${y-19} ${x+18},${y} ${x},${y+19} ${x-18},${y}`,fill:'#61edcb',stroke:'#a7ffe6','stroke-width':2});else{shape('circle',{cx:x,cy:y,r:22,fill:'#5a283f',opacity:.55});shape('path',{d:`M${x-12},${y-12}L${x+12},${y+12} M${x+12},${y-12}L${x-12},${y+12}`,stroke:'#ff8193','stroke-width':7,'stroke-linecap':'round'})}const arrow=shape('text',{x,y:y+37,fill:'#acbfdc','font-size':17,'text-anchor':'middle'});arrow.textContent=o.vx<0?'←':o.vx>0?'→':'·';}
shape('rect',{x:s.paddle_column*80+11,y:7*80+34,width:58,height:13,rx:5,fill:'#b7a4ff'});
$('score').textContent=f.score;$('tick').textContent=`${f.tick} / 256`;$('action').textContent=f.action==null?'—':['Stay','Left','Right'][f.action];$('timeline').value=index;$('timeline').max=Math.max(0,frames.length-1);
$('status').textContent=f.terminated?'Game over · bomb hit':f.truncated?'Game over · time limit':paused?'Paused':'Playing';$('status').style.color=f.terminated?'#ff8193':'#65edcc';
$('logits').textContent=f.logits?'Logits · stay / left / right\n'+f.logits.map(x=>Number(x).toPrecision(5)).join(' / '):'';
$('pause').textContent=paused?'Play':'Pause';}
async function request(path,payload){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!r.ok)throw Error(await r.text());return r.json()}
async function advance(){if(inFlight)return;const f=frames[index];if(f&&(f.terminated||f.truncated)){paused=true;draw();return}if(live){inFlight=true;try{const next=await request('/step',{action:left&&!right?1:right&&!left?2:0});frames.push(next);index=frames.length-1;}catch(e){paused=true;$('status').textContent=e.message;}finally{inFlight=false;if(pendingRestart){pendingRestart=false;$('restart').click();}}}else if(index<frames.length-1)index++;else paused=true;draw()}
function schedule(){clearInterval(timer);timer=setInterval(()=>{if(!paused)advance()},1000/Number($('speed').value))}
$('pause').onclick=()=>{paused=!paused;draw()};$('step').onclick=()=>{paused=true;advance()};$('speed').onchange=schedule;
$('restart').onclick=async()=>{paused=true;if(inFlight){pendingRestart=true;return}if(live){inFlight=true;try{const fresh=await request('/restart',{});frames.splice(0,frames.length,fresh)}finally{inFlight=false;pendingRestart=false}}index=0;draw()};
$('timeline').oninput=()=>{if(!live){paused=true;index=Number($('timeline').value);draw()}};
window.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight',' '].includes(e.key))e.preventDefault();if(e.key==='ArrowLeft'||e.key==='a')left=true;if(e.key==='ArrowRight'||e.key==='d')right=true;if(!e.repeat&&e.key===' '){paused=!paused;draw()}if(!e.repeat&&e.key==='n')$('step').click();if(!e.repeat&&e.key==='r')$('restart').click()});
window.addEventListener('keyup',e=>{if(e.key==='ArrowLeft'||e.key==='a')left=false;if(e.key==='ArrowRight'||e.key==='d')right=false});window.addEventListener('blur',()=>{left=right=false;paused=true;draw()});
$('mode').textContent=live?'Human play · local':'Recorded replay';$('policy').textContent=data.policy_name||'Human';$('selection').textContent=data.selection||'Manual episode';$('help').innerHTML=live?'<kbd>←</kbd> <kbd>→</kbd> move · no key stays · <kbd>Space</kbd> pause · <kbd>N</kbd> step · <kbd>R</kbd> restart. Press Play to begin.':'Recorded states and actions. Play, step, or scrub the timeline to inspect the episode.';$('timeline').hidden=live;draw();schedule();
</script></html>'''


def render_html(record: dict, *, live=False):
    # Even an untrusted policy name cannot break out of the script element.
    payload = json.dumps(record, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    return PAGE.replace('__LIVE__', 'true' if live else 'false').replace('__DATA__', payload)


def write_replay_html(record: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(record), encoding='utf-8')
    return path


class HumanGame:
    def __init__(self, seed=0, save_path=None, config=GameConfig()):
        self.config = config
        self.tape = make_tape(seed, config)
        self.save_path = Path(save_path) if save_path else None
        self.restart()

    def restart(self):
        self.env = Env(self.tape, self.config)
        self.record = {'format': 'ricochet-replay-v1', 'policy_name': 'Human',
                       'selection': 'Manual human episode', 'trajectory': []}
        return self._record(None, None)

    def _record(self, action, result):
        from .evaluate import state_record
        events = [] if result is None else [
            {'kind': e.kind, 'object': e.object._asdict() if e.object else None,
             'paddle_column': e.paddle_column} for e in result.events]
        frame = {'tick': self.env.elapsed_ticks, 'state': state_record(self.env.state),
                 'score': self.env.score, 'action': action, 'logits': None,
                 'reward': 0 if result is None else result.reward,
                 'events': events, 'terminated': self.env.terminated,
                 'truncated': self.env.truncated}
        self.record['trajectory'].append(frame)
        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.save_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.record), encoding='utf-8')
            temporary.replace(self.save_path)
        return frame

    def step(self, action):
        result = self.env.step(action)
        return self._record(action, result)


def make_human_server(seed=0, save_path=None, port=0, config=GameConfig()):
    game = HumanGame(seed, save_path, config)
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, data, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/':
                self._reply(404, b'Not found', 'text/plain')
                return
            with lock:
                data = render_html(game.record, live=True).encode()
            self._reply(200, data, 'text/html; charset=utf-8')

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{self.server.server_port}':
                self._reply(403, b'Cross-origin requests are not accepted', 'text/plain')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(length))
                with lock:
                    if self.path == '/step' and isinstance(request, dict) and set(request) == {'action'}:
                        result = game.step(request['action'])
                    elif self.path == '/restart' and request == {}:
                        result = game.restart()
                    else:
                        raise ValueError('Unknown request')
                self._reply(200, json.dumps(result).encode())
            except (ValueError, RuntimeError, TypeError) as error:
                self._reply(400, str(error).encode(), 'text/plain')

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.game = game
    return server


def play(seed=0, save_path=None, port=0, open_browser=True, config=GameConfig()):
    server = make_human_server(seed, save_path, port, config)
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'Human game: {url}\nPress Ctrl+C here to stop. Space pauses; N steps; R restarts.')
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
