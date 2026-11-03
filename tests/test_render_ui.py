"""Run the shipped browser script in a small DOM with controlled network timing.

The deferred fetches exercise a genuine asynchronous Step/Restart race. This
does not stand in for visual inspection; it verifies the controls' behavior.
"""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ricochet_catch.render import render_html


BUNDLED_NODE = Path("/Users/michal/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE = shutil.which("node") or (str(BUNDLED_NODE) if BUNDLED_NODE.is_file() else None)

HARNESS = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
class Element {
  constructor(){this.children=[];this.attributes={};this.style={};this.value='';this.textContent='';}
  setAttribute(key,value){this.attributes[key]=String(value);}
  appendChild(child){this.children.push(child);return child;}
  replaceChildren(){this.children=[];}
  click(){this.lastClick=this.onclick?.();}
}
const elements = new Map(), listeners = {}, calls = [], timers = new Map();
const element = id => {if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
element('speed').value='5';
const context=vm.createContext({
  document:{getElementById:element,createElementNS:()=>new Element()},
  window:{addEventListener:(name,fn)=>listeners[name]=fn},
  setInterval:(fn,delay)=>{const id=timers.size+1;timers.set(id,{fn,delay});return id;},
  clearInterval:id=>timers.delete(id),
  fetch:(path,options)=>new Promise(resolve=>calls.push({path,body:JSON.parse(options.body),resolve})),
  assert,element,listeners,calls,timers,
});
vm.runInContext(input.script,context);
const flush=async()=>{for(let i=0;i<15;i++)await Promise.resolve();};
const respond=(call,frame)=>call.resolve({ok:true,json:async()=>structuredClone(frame)});
const key=(kind,key)=>listeners[kind]({key,repeat:false,preventDefault(){}});
Object.assign(context,{flush,respond,key,initial:input.initial});
(async()=>{await vm.runInContext('(async()=>{'+input.scenario+'})()',context);})()
  .then(()=>process.stdout.write('ok\n')).catch(error=>{console.error(error);process.exitCode=1;});
"""


def frame(tick=0, *, action=None, score=0, terminated=False, truncated=False):
    return {"tick": tick, "score": score, "action": action, "logits": None,
            "state": {"paddle_column": 3, "objects": [], "remaining_ticks": 256 - tick},
            "terminated": terminated, "truncated": truncated}


def run_browser_script(scenario, *, live=True, frames=None, policy="Human", selection="Manual episode"):
    if NODE is None:
        pytest.skip("Node.js is unavailable for the browser behavior test")
    initial = frame()
    record = {"policy_name": policy, "selection": selection, "trajectory": frames or [initial]}
    html = render_html(record, live=live)
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run([NODE, "-e", HARNESS], input=json.dumps({
        "script": script, "scenario": scenario, "initial": initial,
    }), text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n"


def test_rapid_step_then_restart_waits_and_cannot_append_a_stale_frame():
    run_browser_script(r"""
      assert.equal(element('tick').textContent,'0 / 256');
      element('step').click();
      assert.equal(calls.length,1);
      assert.equal(calls[0].path,'/step');
      element('restart').click();
      element('step').click();
      assert.equal(calls.length,1,'restart must wait for the pending step');
      respond(calls[0],{...initial,tick:1,action:0,score:1});
      await flush();
      assert.equal(calls.length,2);
      assert.equal(calls[1].path,'/restart');
      respond(calls[1],initial);
      await flush();
      assert.equal(element('tick').textContent,'0 / 256');
      assert.equal(element('score').textContent,0);
      assert.equal(frames.length,1);
      assert.equal(index,0);
      assert.equal(inFlight,false);
      assert.equal(paused,true);
      for(const timer of timers.values())timer.fn();
      await flush();
      assert.equal(calls.length,2,'reset must remain paused');
      assert.equal(element('tick').textContent,'0 / 256');
    """)


def test_keys_step_pause_speed_and_terminal_controls():
    run_browser_script(r"""
      assert.equal(element('timeline').hidden,true);
      assert.equal([...timers.values()][0].delay,200);
      element('speed').value='10';element('speed').onchange();
      assert.equal([...timers.values()][0].delay,100);
      key('keydown','ArrowRight');element('step').click();
      assert.equal(calls[0].body.action,2);
      respond(calls[0],{...initial,tick:1,action:2});await flush();
      assert.equal(element('action').textContent,'Right');
      key('keyup','ArrowRight');key('keydown','ArrowLeft');key('keydown','n');
      assert.equal(calls[1].body.action,1);
      respond(calls[1],{...initial,tick:2,action:1});await flush();
      key('keydown','ArrowRight');element('step').click();
      assert.equal(calls[2].body.action,0,'opposing held keys select stay');
      respond(calls[2],{...initial,tick:3,action:0,terminated:true});await flush();
      assert.equal(element('status').textContent,'Game over · bomb hit');
      element('step').click();await flush();
      assert.equal(calls.length,3,'finished episode must not request another step');
      key('keydown',' ');listeners.blur();
      assert.equal(paused,true);
      assert.equal(left,false);assert.equal(right,false);
    """)


def test_repeated_restart_clicks_do_not_reset_the_next_step():
    run_browser_script(r"""
      element('restart').click();
      element('restart').click();
      element('restart').click();
      assert.equal(calls.length,1,'duplicate reset requests should coalesce');
      assert.equal(calls[0].path,'/restart');
      respond(calls[0],initial);await flush();
      assert.equal(pendingRestart,false);
      element('step').click();
      assert.equal(calls.length,2);
      assert.equal(calls[1].path,'/step');
      respond(calls[1],{...initial,tick:1,action:0});await flush();
      assert.equal(calls.length,2,'the next step must not trigger another reset');
      assert.equal(element('tick').textContent,'1 / 256');
      assert.equal(frames.length,2);
    """)


def test_replay_scrubbing_uses_recorded_state_and_preserves_placeholder_text():
    run_browser_script(r"""
      assert.equal(element('policy').textContent,'policy __LIVE__ __DATA__');
      assert.equal(element('selection').textContent,'selection __DATA__ __LIVE__');
      element('timeline').value='1';element('timeline').oninput();
      assert.equal(element('tick').textContent,'256 / 256');
      assert.equal(element('score').textContent,4);
      assert.equal(element('status').textContent,'Game over · time limit');
      element('restart').click();await flush();
      assert.equal(element('tick').textContent,'0 / 256');
      element('step').click();await flush();
      assert.equal(element('tick').textContent,'256 / 256');
      assert.equal(calls.length,0,'replay never calls the simulation server');
    """, live=False, frames=[frame(), frame(256, action=0, score=4, truncated=True)],
        policy="policy __LIVE__ __DATA__", selection="selection __DATA__ __LIVE__")
