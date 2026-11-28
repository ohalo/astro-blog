(function () {
  'use strict';

  const COLS = 8;
  const ROWS = 8;
  const MAX_LV = 5;
  const TRAY_SIZE = 6;
  const RECRUIT_COST = 3;
  const WAVE_INTERVAL = 11;
  const MAX_HP = 3;

  const BASE = {
    '刀': { dmg: 9, range: 1.25, rate: 0.75 },
    '枪': { dmg: 13, range: 2.10, rate: 1.05, pierce: 2 },
    '弓': { dmg: 7, range: 3.10, rate: 0.62 },
    '骑': { dmg: 11, range: 1.55, rate: 0.48, splash: 0.5 }
  };

  // 武将按兵器分三类：刀 = 近战、枪 = 中程穿刺、弓 = 远程狙击。
  // wtype 决定棋子上的兵器标记、弹道外观和属性面板里的兵种说明。
  const HEROES = {
    '赵云': { chars: ['赵', '云'], wtype: '枪', dmg: 26, range: 2.60, rate: 0.50, skill: 'pierceAll', skillText: '龙胆突刺 · 贯穿整条路径' },
    '关羽': { chars: ['关', '羽'], wtype: '刀', dmg: 32, range: 1.35, rate: 0.90, skill: 'splash', skillText: '青龙斩 · 溅射半数伤害' },
    '张飞': { chars: ['张', '飞'], wtype: '枪', dmg: 36, range: 1.80, rate: 1.00, skill: 'slow', skillText: '丈八蛇矛 · 命中减速' },
    '黄忠': { chars: ['黄', '忠'], wtype: '弓', dmg: 22, range: 3.40, rate: 0.80, skill: 'snipe', skillText: '落日弓 · 优先狙击高血量敌军' },
    '刘备': { chars: ['刘', '备'], wtype: '刀', dmg: 18, range: 1.50, rate: 0.85, skill: 'heal', skillText: '双股剑 · 每 8 次出手回护阿斗' },
    '马超': { chars: ['马', '超'], wtype: '枪', dmg: 19, range: 1.60, rate: 0.50, skill: 'pierce2', skillText: '虎头枪 · 一枪穿透两名敌军' },
    '姜维': { chars: ['姜', '维'], wtype: '枪', dmg: 24, range: 2.40, rate: 0.70, skill: 'splash', skillText: '麒麟枪 · 溅射半数伤害' },
    '黄盖': { chars: ['黄', '盖'], wtype: '刀', dmg: 28, range: 1.10, rate: 1.10, skill: 'slow', skillText: '铁背刀 · 命中减速' }
  };

  const WEAPON_NAMES = {
    '赵云': '龙胆枪', '关羽': '青龙刀', '张飞': '丈八蛇矛', '黄忠': '落日弓',
    '刘备': '双股剑', '马超': '虎头枪', '姜维': '麒麟枪', '黄盖': '铁背刀'
  };
  const WEAPON_TYPES = {
    '刀': { label: '近战 · 刀', desc: '贴身快攻，射程短但攻速快' },
    '枪': { label: '中程 · 枪', desc: '一寸长一寸强，穿刺路径上的敌军' },
    '弓': { label: '远程 · 弓', desc: '百步穿杨，射程最远' }
  };
  const QUALITIES = [
    { name: '白', mult: 1.10 }, { name: '绿', mult: 1.25 },
    { name: '蓝', mult: 1.40 }, { name: '紫', mult: 1.60 }, { name: '金', mult: 1.85 }
  ];

  const RECRUIT_TABLE = [
    { key: '刀', type: 'base', w: 12 }, { key: '枪', type: 'base', w: 10 },
    { key: '弓', type: 'base', w: 10 }, { key: '骑', type: 'base', w: 9 },
    { key: '赵', type: 'char', w: 8 }, { key: '云', type: 'char', w: 8 },
    { key: '关', type: 'char', w: 6 }, { key: '羽', type: 'char', w: 6 },
    { key: '张', type: 'char', w: 5 }, { key: '飞', type: 'char', w: 5 },
    { key: '黄', type: 'char', w: 5 }, { key: '忠', type: 'char', w: 5 },
    { key: '刘', type: 'char', w: 4 }, { key: '备', type: 'char', w: 4 },
    { key: '马', type: 'char', w: 4 }, { key: '超', type: 'char', w: 4 },
    { key: '姜', type: 'char', w: 3 }, { key: '维', type: 'char', w: 3 },
    { key: '盖', type: 'char', w: 2 }, { key: '平', type: 'char', w: 2 },
    { key: 'shovel', type: 'shovel', w: 5 }
  ];
  const TABLE_TOTAL = RECRUIT_TABLE.reduce((s, r) => s + r.w, 0);

  const $ = (id) => document.getElementById(id);
  const el = {
    playerGrid: $('playerGrid'), aiGrid: $('aiGrid'),
    tray: $('tray'), ghost: $('ghost'), overlay: $('overlay'), panel: $('panel'),
    mantou: $('mantouCount'), gold: $('goldCount'), wave: $('waveLabel'), time: $('timeLabel'),
    myHp: $('myHp'), aiHp: $('aiHp'), myAdou: $('myAdou'), aiAdou: $('aiAdou'),
    recruit: $('recruitBtn'), hint: $('hint'),
    merchant: $('merchant'), merchantCards: $('merchantCards'),
    merchantTimer: $('merchantTimer'), merchantGold: $('merchantGold'),
    merchantBtn: $('merchantBtn'), pauseBtn: $('pauseBtn'),
    pausePanel: $('pausePanel'), resumeBtn: $('resumeBtn'),
    waveBar: $('waveBar'), buffs: $('buffs'),
    heroCard: $('heroCard'), heroCardBody: $('heroCardBody'), heroCardClose: $('heroCardClose'),
    rangeRing: $('rangeRing')
  };

  const SAVE_KEY = 'zyad_save_v2';
  function loadSave() {
    let raw = null;
    try { raw = JSON.parse(localStorage.getItem(SAVE_KEY) || 'null'); } catch (e) { raw = null; }
    if (!raw) raw = { weapons: [], equipped: {}, best: Number(localStorage.getItem('zyad_best') || 0), nextId: 1 };
    if (!Array.isArray(raw.weapons)) raw.weapons = [];
    if (!raw.equipped) raw.equipped = {};
    if (typeof raw.best !== 'number') raw.best = 0;
    if (typeof raw.nextId !== 'number') raw.nextId = 1;
    if (!raw.weapons.length) {
      raw.weapons.push({ id: raw.nextId++, hero: '赵云', q: 0 });
      raw.equipped['赵云'] = raw.weapons[0].id;
    }
    return raw;
  }
  let save = loadSave();
  function persist() { try { localStorage.setItem(SAVE_KEY, JSON.stringify(save)); } catch (e) {} }
  function weaponName(w) { return QUALITIES[w.q].name + '·' + WEAPON_NAMES[w.hero]; }
  function dropWeapon(win) {
    const heroes = Object.keys(WEAPON_NAMES);
    const hero = heroes[Math.floor(Math.random() * heroes.length)];
    let q = win ? Math.floor(Math.random() * 4) : Math.floor(Math.random() * 3);
    if (win && Math.random() < 0.06) q = 4;
    const w = { id: save.nextId++, hero: hero, q: q };
    save.weapons.push(w);
    persist();
    return w;
  }

  const state = {
    running: false, merchantOpen: false, userPaused: false,
    time: 0, wave: 0, waveTimer: 2.5,
    lastFrame: 0, player: null, ai: null,
    projectiles: [], hintTimer: 0, hudTimer: 0,
    merchantTime: 0, merchantChoices: [], inspect: null
  };
  function isFrozen() { return state.merchantOpen || state.userPaused; }

  function generatePath() {
    const pts = [];
    let col = 2 + Math.floor(Math.random() * (COLS - 4));
    pts.push({ row: -1, col: col });
    pts.push({ row: 0, col: col });
    for (let row = 1; row < ROWS; row++) {
      const r = Math.random();
      let next = col;
      if (r < 0.32 && col < COLS - 2) next = col + 1;
      else if (r < 0.64 && col > 1) next = col - 1;
      pts.push({ row: row, col: col });
      const step = next > col ? 1 : -1;
      while (col !== next) {
        col += step;
        pts.push({ row: row, col: col });
      }
    }
    pts.push({ row: ROWS, col: col });
    return pts;
  }

  function makeSide(isAI) {
    const side = {
      isAI: isAI, mantou: 18, gold: 0, hp: MAX_HP,
      dmgBuffUntil: 0, dmgBuff: 1,
      rateBuffUntil: 0, rateBuff: 1,
      incomeUntil: 0, incomeMult: 1,
      cells: new Array(COLS * ROWS).fill(null),
      blocked: new Set(), pathCells: new Set(), pts: [],
      tray: new Array(TRAY_SIZE).fill(null),
      enemies: [], spawnQueue: [], incomeTimer: 0,
      aiRecruit: 1.5, aiPlace: 0.5, aiSynth: 1.5,
      hasZhao: false,
      gridEl: isAI ? el.aiGrid : el.playerGrid,
      cellEls: []
    };
    side.cellLayer = side.gridEl.querySelector('.cell-layer');
    side.enemyLayer = side.gridEl.querySelector('.enemy-layer');
    side.fxLayer = side.gridEl.querySelector('.fx-layer');
    side.pathSvg = side.gridEl.querySelector('.path-svg polyline');

    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('class', 'path-svg');
    svg.setAttribute('viewBox', '0 0 ' + COLS + ' ' + ROWS);
    svg.setAttribute('preserveAspectRatio', 'none');
    const poly = document.createElementNS(ns, 'polyline');
    svg.appendChild(poly);
    side.gridEl.insertBefore(svg, side.enemyLayer);
    side.pathSvg = poly;

    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS; col++) {
        const c = document.createElement('div');
        c.className = 'cell';
        c.style.left = (col / COLS * 100) + '%';
        c.style.top = (row / ROWS * 100) + '%';
        c.style.width = (100 / COLS) + '%';
        c.style.height = (100 / ROWS) + '%';
        c.dataset.index = String(row * COLS + col);
        side.cellLayer.appendChild(c);
        side.cellEls.push(c);
      }
    }
    return side;
  }

  function newBoard(side, pts) {
    side.pts = pts.map((p) => side.isAI
      ? { row: ROWS - 1 - p.row, col: COLS - 1 - p.col }
      : { row: p.row, col: p.col });
    side.pathCells = new Set();
    side.pts.forEach((p) => {
      if (p.row >= 0 && p.row < ROWS) side.pathCells.add(p.row * COLS + p.col);
    });
    side.pathSvg.setAttribute('points', side.pts.map((p) => (p.col + 0.5) + ',' + (p.row + 0.5)).join(' '));
    side.blocked.clear();
    const candidates = [];
    for (let i = 0; i < side.cells.length; i++) {
      if (!side.pathCells.has(i)) candidates.push(i);
    }
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = candidates[i]; candidates[i] = candidates[j]; candidates[j] = t;
    }
    for (let i = 0; i < 5 && i < candidates.length; i++) side.blocked.add(candidates[i]);
  }

  function randomItem() {
    let r = Math.random() * TABLE_TOTAL;
    for (const entry of RECRUIT_TABLE) {
      r -= entry.w;
      if (r <= 0) return { type: entry.type, key: entry.key };
    }
    return { type: 'base', key: '刀' };
  }

  function itemLabel(item) {
    if (!item) return '';
    return item.type === 'shovel' ? '铲' : item.key;
  }

  function showHint(text) {
    el.hint.textContent = text;
    state.hintTimer = 2.2;
  }

  function shake(node) {
    node.classList.remove('shake');
    void node.offsetWidth;
    node.classList.add('shake');
  }

  function renderCells(side) {
    for (let i = 0; i < side.cells.length; i++) {
      const cell = side.cellEls[i];
      const unit = side.cells[i];
      let sig = unit ? (unit.kind + '|' + unit.key + '|' + unit.lv + (unit.span2 ? '|wide' : ''))
        : (side.blocked.has(i) ? 'blocked' : (side.pathCells.has(i) ? 'path' : ''));
      if (cell.dataset.sig === sig) continue;
      cell.dataset.sig = sig;
      const inRange = state.inspect && !side.isAI && state.inspect.rangeSet.has(i);
      cell.className = 'cell' + (side.pathCells.has(i) ? ' path' : '') +
        (side.blocked.has(i) ? ' blocked' : '') + (inRange ? ' in-range' : '');
      cell.style.zIndex = '';
      if (!unit) { cell.innerHTML = ''; continue; }
      if (unit.kind === 'span') { cell.classList.add('span-part'); cell.innerHTML = ''; continue; }
      // 字牌合成武将时签名会变，先清空旧节点，避免字与武将重叠。
      cell.innerHTML = '';
      const isHero = unit.kind === 'hero';
      const isChar = unit.kind === 'char';
      const node = document.createElement('div');
      node.className = 'unit ' + (isHero ? 'hero gold' : (isChar ? 'gold' : '')) + (unit.span2 ? ' wide' : '') +
        (state.inspect && !side.isAI && state.inspect.idx === i ? ' inspecting' : '');
      node.innerHTML = (isHero ? '<span class="glow"></span>' : '') + '<span>' + unit.key + '</span>' +
        (isHero ? '<span class="wpn-mark w-' + HEROES[unit.key].wtype + '">' + HEROES[unit.key].wtype + '</span>' : '') +
        (unit.lv > 1 ? '<span class="lv">' + unit.lv + '</span>' : '');
      cell.appendChild(node);
      if (unit.span2) cell.style.zIndex = '3';
    }
  }

  function renderTray(side) {
    if (side.isAI) return;
    const sig = side.tray.map((t) => t ? t.type + ':' + t.key : '-').join('|');
    if (el.tray.dataset.sig === sig) return;
    el.tray.dataset.sig = sig;
    el.tray.innerHTML = '';
    for (let i = 0; i < TRAY_SIZE; i++) {
      const item = side.tray[i];
      const slot = document.createElement('div');
      slot.className = 'slot' + (item ? ' filled' : '') +
        (item && item.type === 'char' ? ' gold' : '') + (item && item.type === 'shovel' ? ' shovel' : '');
      slot.dataset.index = String(i);
      if (item) {
        slot.textContent = itemLabel(item);
        if (item.type === 'shovel') {
          const tag = document.createElement('span');
          tag.className = 'slot-tag';
          tag.textContent = '清石';
          slot.appendChild(tag);
        }
      }
      el.tray.appendChild(slot);
    }
    if (selected && selected.source === 'tray' && el.tray.children[selected.index]) {
      el.tray.children[selected.index].classList.add('selected');
    }
  }

  function updateBuffs() {
    const p = state.player;
    const chips = [];
    if (state.time < p.dmgBuffUntil) chips.push('神兵符 ' + Math.ceil(p.dmgBuffUntil - state.time) + 's');
    if (state.time < p.rateBuffUntil) chips.push('攻速符 ' + Math.ceil(p.rateBuffUntil - state.time) + 's');
    if (state.time < p.incomeUntil) chips.push('农民 ' + Math.ceil(p.incomeUntil - state.time) + 's');
    if (p.hasZhao) chips.push('赵云羁绊');
    const sig = chips.join('|');
    if (el.buffs.dataset.sig === sig) return;
    el.buffs.dataset.sig = sig;
    el.buffs.innerHTML = chips.map((c) => '<span class="buff-chip">' + c + '</span>').join('');
  }

  function updateHud() {
    el.mantou.textContent = String(state.player.mantou);
    el.gold.textContent = String(state.player.gold);
    el.merchantGold.textContent = String(state.player.gold);
    el.wave.textContent = '第 ' + Math.max(1, state.wave) + ' 波';
    const sec = Math.floor(state.time);
    el.time.textContent = String(Math.floor(sec / 60)).padStart(2, '0') + ':' + String(sec % 60).padStart(2, '0');
    el.myHp.textContent = '●'.repeat(state.player.hp) + '○'.repeat(MAX_HP - state.player.hp);
    el.aiHp.textContent = '●'.repeat(state.ai.hp) + '○'.repeat(MAX_HP - state.ai.hp);
    el.myAdou.classList.toggle('low', state.player.hp === 1);
    el.aiAdou.classList.toggle('low', state.ai.hp === 1);
    el.recruit.disabled = !state.running || isFrozen() ||
      state.player.mantou < RECRUIT_COST || state.player.tray.every(Boolean);
    el.merchantBtn.disabled = !state.running || isFrozen();
    el.waveBar.style.width = Math.min(100, (1 - state.waveTimer / WAVE_INTERVAL) * 100) + '%';
    updateBuffs();
    if (state.inspect) fillHeroCard(state.inspect.idx, state.player.cells[state.inspect.idx]);
  }

  function startWave() {
    state.wave++;
    for (const side of [state.player, state.ai]) {
      const count = 2 + Math.min(state.wave, 12);
      for (let i = 0; i < count; i++) side.spawnQueue.push({ delay: i * 0.72, boss: false });
      if (state.wave % 5 === 0) side.spawnQueue.push({ delay: count * 0.72 + 0.6, boss: true });
    }
    state.player.gold += 3;
    showHint('第 ' + state.wave + ' 波来袭' + (state.wave % 5 === 0 ? ' · 敌军首领出现！' : ''));
    updateHud();
  }

  function spawnEnemy(side, boss) {
    const wave = state.wave;
    const hp = 22 * Math.pow(1.16, wave - 1) * (boss ? 6 : 1);
    const node = document.createElement('div');
    node.className = 'enemy' + (boss ? ' boss' : '');
    node.textContent = boss ? '王' : (Math.random() < 0.5 ? '贼' : '寇');
    node.innerHTML += '<span class="enemy-hp"><i style="width:100%"></i></span>';
    side.enemyLayer.appendChild(node);
    const enemy = {
      s: 0, row: side.pts[0].row, col: side.pts[0].col,
      hp: hp, maxHp: hp,
      speed: Math.min(1.2, 0.55 + 0.025 * wave) * (boss ? 0.7 : 1),
      boss: boss, slowTimer: 0, node: node,
      bar: node.querySelector('.enemy-hp i')
    };
    side.enemies.push(enemy);
    placeEnemy(side, enemy);
  }

  function pathPos(side, enemy) {
    const pts = side.pts;
    const s = Math.max(0, Math.min(pts.length - 1.0001, enemy.s));
    const i = Math.floor(s);
    const t = s - i;
    const a = pts[i];
    const b = pts[Math.min(i + 1, pts.length - 1)];
    return { row: a.row + (b.row - a.row) * t, col: a.col + (b.col - a.col) * t };
  }

  function enemyXY(side, enemy) {
    const w = side.gridEl.clientWidth / COLS;
    const h = side.gridEl.clientHeight / ROWS;
    return { x: (enemy.col + 0.5) * w - 12, y: (enemy.row + 0.5) * h - 12 };
  }

  function placeEnemy(side, enemy) {
    const p = pathPos(side, enemy);
    enemy.row = p.row;
    enemy.col = p.col;
    const pos = enemyXY(side, enemy);
    enemy.node.style.transform = 'translate(' + pos.x + 'px,' + pos.y + 'px)';
    if (enemy.bar) enemy.bar.style.width = Math.max(0, enemy.hp / enemy.maxHp * 100) + '%';
  }

  function flashingCells(side, indices) {
    indices.forEach((idx) => {
      const c = side.cellEls[idx];
      if (!c) return;
      c.style.background = 'rgba(168,121,27,.28)';
      setTimeout(() => { c.style.background = ''; }, 600);
    });
  }

  function checkHero(side) {
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS - 1; col++) {
        const i = row * COLS + col;
        const a = side.cells[i];
        const b = side.cells[i + 1];
        if (!a || !b || a.kind !== 'char' || b.kind !== 'char') continue;
        const name = a.key + b.key;
        if (!HEROES[name]) continue;
        side.cells[i] = { kind: 'hero', key: name, lv: 1, span2: true, atk: 0, shots: 0 };
        side.cells[i + 1] = { kind: 'span', owner: i };
        if (name === '赵云') side.hasZhao = true;
        flashingCells(side, [i, i + 1]);
        if (!side.isAI) showHint(name + ' 现身！' + (name === '赵云' ? ' 全军攻势 +30%' : ' 占据两格出战'));
        return true;
      }
    }
    return false;
  }

  function damageOf(unit, side) {
    const spec = unit.kind === 'hero' ? HEROES[unit.key] : BASE[unit.key];
    if (!spec) return 0;
    let mult = Math.pow(1.65, (unit.lv || 1) - 1);
    if (side.hasZhao) mult *= 1.3;
    if (unit.kind === 'hero' && !side.isAI) {
      const wid = save.equipped[unit.key];
      const weapon = save.weapons.find((w) => w.id === wid && w.hero === unit.key);
      if (weapon) mult *= QUALITIES[weapon.q].mult;
    }
    if (state.time < side.dmgBuffUntil) mult *= side.dmgBuff;
    return spec.dmg * mult;
  }

  // ---- 点击武将：属性卡 + 实际攻击范围高亮 ----
  function clearInspect() {
    if (!state.inspect) return;
    state.inspect = null;
    el.heroCard.classList.add('hidden');
    el.rangeRing.classList.add('hidden');
    // 空格签名本身是空串，必须用非零哨兵才能让 renderCells 重建。
    state.player.cellEls.forEach((c) => { c.dataset.sig = 'refresh'; });
    renderCells(state.player);
  }

  function fillHeroCard(idx, unit) {
    const spec = HEROES[unit.key];
    const type = WEAPON_TYPES[spec.wtype];
    const rate = spec.rate * (state.time < state.player.rateBuffUntil ? state.player.rateBuff : 1);
    const wid = save.equipped[unit.key];
    const weapon = save.weapons.find((w) => w.id === wid && w.hero === unit.key);
    el.heroCardBody.innerHTML =
      '<div class="hc-head"><span class="hc-name">' + unit.key + '</span>' +
      '<span class="hc-type w-' + spec.wtype + '">' + spec.wtype + '</span></div>' +
      '<div class="hc-type-label">' + type.label + ' · ' + type.desc + '</div>' +
      '<div class="hc-stats">' +
      '<div><b>' + spec.range.toFixed(1) + '</b><span>射程 / 格</span></div>' +
      '<div><b>' + rate.toFixed(2) + 's</b><span>攻速 / 每次</span></div>' +
      '<div><b>' + (1 / rate).toFixed(1) + '</b><span>每秒出手</span></div>' +
      '<div><b>' + Math.round(damageOf(unit, state.player)) + '</b><span>当前伤害</span></div>' +
      '</div>' +
      '<div class="hc-skill">' + spec.skillText + '</div>' +
      '<div class="hc-weapon">专属兵器：' + (weapon ? weaponName(weapon) : '未装备') + '</div>' +
      '<div class="hc-tip">高亮格为可攻击范围 · 点别处收起</div>';
  }

  function refreshInspectView() {
    const ins = state.inspect;
    if (!ins) return;
    const unit = state.player.cells[ins.idx];
    if (!unit || unit.kind !== 'hero') { clearInspect(); return; }
    const spec = HEROES[unit.key];
    const cw = state.player.gridEl.clientWidth / COLS;
    const ch = state.player.gridEl.clientHeight / ROWS;
    const radius = spec.range * cw;
    const cx = (ins.idx % COLS + 1) * cw;
    const cy = (Math.floor(ins.idx / COLS) + 0.5) * ch;
    el.rangeRing.style.width = el.rangeRing.style.height = (radius * 2) + 'px';
    el.rangeRing.style.left = (cx - radius) + 'px';
    el.rangeRing.style.top = (cy - radius) + 'px';
    el.rangeRing.classList.remove('hidden');
    el.heroCard.classList.remove('hidden');
    fillHeroCard(ins.idx, unit);
    state.player.cellEls.forEach((c) => { c.dataset.sig = 'refresh'; });
    renderCells(state.player);
  }

  function setInspect(idx) {
    if (state.inspect && state.inspect.idx === idx) { clearInspect(); return; }
    const unit = state.player.cells[idx];
    if (!unit || unit.kind !== 'hero') return;
    const spec = HEROES[unit.key];
    const row = Math.floor(idx / COLS);
    const col = idx % COLS + 1; // 武将占两格，从身体中心计算射程
    const rangeSet = new Set();
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const dx = c + 0.5 - col;
        const dy = r + 0.5 - (row + 0.5);
        if (Math.sqrt(dx * dx + dy * dy) <= spec.range + 1e-6) rangeSet.add(r * COLS + c);
      }
    }
    state.inspect = { idx: idx, rangeSet: rangeSet };
    refreshInspectView();
    showHint(unit.key + ' · ' + WEAPON_TYPES[spec.wtype].label + ' · 射程 ' +
      spec.range.toFixed(1) + ' 格 · 攻速 ' + spec.rate.toFixed(2) + 's');
  }

  function fire(side, unit, row, col, target, skill) {
    const w = side.gridEl.clientWidth / COLS;
    const h = side.gridEl.clientHeight / ROWS;
    const pos = enemyXY(side, target);
    const mark = unit.kind === 'hero' ? HEROES[unit.key].wtype : unit.key;
    const node = document.createElement('div');
    node.className = 'fx dart w-' + mark + (unit.kind === 'hero' ? ' hero' : '');
    node.dataset.mark = mark;
    side.fxLayer.appendChild(node);
    const startCol = col + (unit.span2 ? 0.5 : 0);
    state.projectiles.push({
      side: side, node: node,
      x1: (startCol + 0.5) * w - 7, y1: (row + 0.5) * h - 7,
      x2: pos.x + 5, y2: pos.y + 5,
      t: 0, dur: 0.16, unit: unit, target: target, skill: skill || ''
    });
    const unitNode = side.cellEls[row * COLS + col].querySelector('.unit');
    if (unitNode) {
      unitNode.classList.remove('attack');
      void unitNode.offsetWidth;
      unitNode.classList.add('attack');
    }
  }

  function applyHit(side, target, amount, skill) {
    if (!side.enemies.includes(target)) return;
    target.hp -= amount;
    const floater = document.createElement('div');
    floater.className = 'float';
    floater.textContent = '-' + Math.round(amount);
    const pos = enemyXY(side, target);
    floater.style.left = (pos.x + 12) + 'px';
    floater.style.top = (pos.y + 12) + 'px';
    side.fxLayer.appendChild(floater);
    setTimeout(() => floater.remove(), 500);
    if (skill === 'slow') target.slowTimer = 1.8;
    if (target.hp <= 0) {
      side.enemies.splice(side.enemies.indexOf(target), 1);
      target.node.remove();
      side.mantou += target.boss ? 10 : 2;
      side.gold += target.boss ? 5 : 1;
    }
  }

  function updateProjectiles(dt) {
    for (let i = state.projectiles.length - 1; i >= 0; i--) {
      const p = state.projectiles[i];
      p.t += dt;
      const k = Math.min(1, p.t / p.dur);
      p.node.style.transform = 'translate(' + (p.x1 + (p.x2 - p.x1) * k) + 'px,' +
        (p.y1 + (p.y2 - p.y1) * k) + 'px)';
      if (k < 1) continue;
      p.node.remove();
      state.projectiles.splice(i, 1);
      const side = p.side;
      const dmg = damageOf(p.unit, side);
      if (p.skill === 'pierceAll') {
        side.enemies.slice().sort((a, b) => a.s - b.s).forEach((v) => applyHit(side, v, dmg, ''));
      } else {
        applyHit(side, p.target, dmg, p.skill);
        if (p.skill === 'splash' || (p.unit.kind === 'hero' && HEROES[p.unit.key].skill === 'splash')) {
          for (const other of side.enemies.slice()) {
            if (other !== p.target && Math.abs(other.s - p.target.s) < 0.5) applyHit(side, other, dmg * 0.5, '');
          }
        }
      }
      if (p.unit.kind === 'hero' && p.unit.key === '刘备') {
        p.unit.shots++;
        if (p.unit.shots % 8 === 0 && side.hp < MAX_HP) {
          side.hp++;
          if (!side.isAI) showHint('刘备回护 · 阿斗恢复 1 点');
        }
      }
    }
  }

  function updateAttacks(side, dt) {
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS; col++) {
        const unit = side.cells[row * COLS + col];
        if (!unit || unit.kind === 'char' || unit.kind === 'span') continue;
        const spec = unit.kind === 'hero' ? HEROES[unit.key] : BASE[unit.key];
        if (!spec) continue;
        unit.atk -= dt;
        if (unit.atk > 0) continue;
        // 两格武将从身体中心开火，和高亮出来的射程圈保持一致。
        const originCol = col + (unit.span2 ? 0.5 : 0);
        const candidates = side.enemies.filter((e) => {
          const dx = originCol - e.col;
          const dy = row - e.row;
          return Math.sqrt(dx * dx + dy * dy) <= spec.range;
        });
        if (!candidates.length) { unit.atk = 0; continue; }
        candidates.sort((a, b) => spec.skill === 'snipe' ? b.hp - a.hp : b.s - a.s);
        const target = candidates[0];
        unit.atk = spec.rate * (state.time < side.rateBuffUntil ? side.rateBuff : 1);
        unit.shots = (unit.shots || 0) + 1;
        let skill = spec.skill || '';
        if (unit.kind === 'base' && spec.pierce && unit.shots % 3 === 0) skill = 'pierce2';
        if (skill === 'pierce2') {
          candidates.slice(0, spec.pierce).forEach((v) => applyHit(side, v, damageOf(unit, side), ''));
          const node = side.cellEls[row * COLS + col].querySelector('.unit');
          if (node) { node.classList.remove('attack'); void node.offsetWidth; node.classList.add('attack'); }
        } else {
          fire(side, unit, row, col, target, skill);
        }
      }
    }
  }

  function updateSide(side, dt) {
    side.incomeTimer += dt;
    while (side.incomeTimer >= 2) {
      side.incomeTimer -= 2;
      side.mantou += state.time < side.incomeUntil ? side.incomeMult : 1;
    }
    for (let i = side.spawnQueue.length - 1; i >= 0; i--) {
      const q = side.spawnQueue[i];
      q.delay -= dt;
      if (q.delay <= 0) {
        spawnEnemy(side, q.boss);
        side.spawnQueue.splice(i, 1);
      }
    }
    for (const enemy of side.enemies.slice()) {
      if (enemy.slowTimer > 0) enemy.slowTimer -= dt;
      enemy.s += enemy.speed * dt * (enemy.slowTimer > 0 ? 0.5 : 1);
      if (enemy.s >= side.pts.length - 1) {
        side.enemies.splice(side.enemies.indexOf(enemy), 1);
        enemy.node.remove();
        side.hp = Math.max(0, side.hp - (enemy.boss ? 2 : 1));
        const adou = side.isAI ? el.aiAdou : el.myAdou;
        adou.classList.remove('hurt');
        void adou.offsetWidth;
        adou.classList.add('hurt');
        if (!side.isAI) showHint('阿斗受创！剩余 ' + side.hp + ' 点');
      } else {
        placeEnemy(side, enemy);
      }
    }
    updateAttacks(side, dt);
    if (side.isAI) updateAI(side, dt);
  }

  function validSpots(side) {
    const spots = [];
    for (let i = 0; i < side.cells.length; i++) {
      if (side.cells[i] || side.blocked.has(i) || side.pathCells.has(i)) continue;
      let d = 99;
      side.pathCells.forEach((pi) => {
        const dist = Math.abs(Math.floor(i / COLS) - Math.floor(pi / COLS)) + Math.abs(i % COLS - pi % COLS);
        if (dist < d) d = dist;
      });
      spots.push({ idx: i, d: d });
    }
    spots.sort((a, b) => (a.d - b.d) + (Math.random() - 0.5) * 0.5);
    return spots.map((s) => s.idx);
  }

  function placeAt(side, idx, item) {
    if (item.type === 'char') side.cells[idx] = { kind: 'char', key: item.key };
    else side.cells[idx] = { kind: 'base', key: item.key, lv: 1, atk: 0, shots: 0 };
    checkHero(side);
    return true;
  }

  function aiPlace(side, item) {
    if (item.type === 'shovel') {
      const it = side.blocked.values().next();
      if (!it.done) side.blocked.delete(it.value);
      return !it.done;
    }
    if (item.type === 'char') {
      for (let i = 0; i < side.cells.length; i++) {
        const u = side.cells[i];
        if (!u || u.kind !== 'char') continue;
        const row = Math.floor(i / COLS);
        for (const name of Object.keys(HEROES)) {
          const parts = HEROES[name].chars;
          if (u.key === parts[0] && (i + 1 < side.cells.length) && Math.floor((i + 1) / COLS) === row &&
              !side.cells[i + 1] && !side.blocked.has(i + 1) && !side.pathCells.has(i + 1)) {
            return placeAt(side, i + 1, item);
          }
          if (u.key === parts[1] && i > 0 && Math.floor((i - 1) / COLS) === row &&
              !side.cells[i - 1] && !side.blocked.has(i - 1) && !side.pathCells.has(i - 1)) {
            return placeAt(side, i - 1, item);
          }
        }
      }
    }
    const spots = validSpots(side);
    return spots.length ? placeAt(side, spots[0], item) : false;
  }

  function updateAI(side, dt) {
    side.aiRecruit -= dt;
    if (side.aiRecruit <= 0) {
      side.aiRecruit = 2.8 + Math.random() * 1.4;
      const slot = side.tray.indexOf(null);
      if (slot >= 0 && side.mantou >= RECRUIT_COST) {
        side.mantou -= RECRUIT_COST;
        side.tray[slot] = randomItem();
      }
    }
    side.aiPlace -= dt;
    if (side.aiPlace <= 0) {
      side.aiPlace = 1.4 + Math.random() * 0.6;
      const slot = side.tray.findIndex((t) => t !== null);
      if (slot >= 0 && aiPlace(side, side.tray[slot])) side.tray[slot] = null;
    }
    side.aiSynth -= dt;
    if (side.aiSynth <= 0) {
      side.aiSynth = 2.6;
      const seen = {};
      for (let i = 0; i < side.cells.length; i++) {
        const u = side.cells[i];
        if (!u || u.kind !== 'base') continue;
        const k = u.key + '-' + u.lv;
        if (seen[k] !== undefined && u.lv < MAX_LV) {
          const j = seen[k];
          side.cells[j] = { kind: 'base', key: u.key, lv: u.lv + 1, atk: 0, shots: 0 };
          side.cells[i] = null;
          break;
        }
        seen[k] = i;
      }
    }
  }

  const MERCHANT_ITEMS = [
    {
      id: 'meteor', name: '陨石', icon: '陨', price: 6, desc: '天火焚敌，全场敌军重创',
      apply: function (side) {
        if (!side.enemies.length) return '场上无敌军，陨石落空';
        side.enemies.slice().forEach((e) => applyHit(side, e, 45 + state.wave * 8, ''));
        return '陨石天降 · 敌军重创';
      }
    },
    {
      id: 'shenbing', name: '神兵符', icon: '符', price: 7, desc: '12 秒全军伤害 +50%',
      apply: function (side) { side.dmgBuffUntil = state.time + 12; side.dmgBuff = 1.5; return '神兵符 · 伤害 +50%'; }
    },
    {
      id: 'yantai', name: '砚台', icon: '砚', price: 4, desc: '随机一名士兵提升一级',
      apply: function (side) {
        const pool = [];
        for (let i = 0; i < side.cells.length; i++) {
          const u = side.cells[i];
          if (u && u.kind === 'base' && u.lv < MAX_LV) pool.push(i);
        }
        if (!pool.length) return '没有可升级的士兵';
        const idx = pool[Math.floor(Math.random() * pool.length)];
        side.cells[idx].lv++;
        renderCells(side);
        return '砚台点化 · ' + side.cells[idx].key + ' 升至 ' + side.cells[idx].lv + ' 级';
      }
    },
    { id: 'baozi', name: '包子', icon: '包', price: 3, desc: '立即获得 10 个馒头',
      apply: function (side) { side.mantou += 10; return '饱食一顿 · 馒头 +10'; } },
    {
      id: 'maobi', name: '毛笔', icon: '笔', price: 5, desc: '改写一字，凑齐武将姓名',
      apply: function (side) {
        const chars = [];
        for (let i = 0; i < side.cells.length; i++) {
          if (side.cells[i] && side.cells[i].kind === 'char') chars.push(i);
        }
        if (chars.length >= 2) {
          const a = chars[Math.floor(Math.random() * chars.length)];
          const others = chars.filter((i) => i !== a);
          const b = others[Math.floor(Math.random() * others.length)];
          if (Math.floor(a / COLS) === Math.floor(b / COLS) && Math.abs(a - b) === 1) {
            const names = Object.keys(HEROES);
            const name = names[Math.floor(Math.random() * names.length)];
            side.cells[a].key = name[0];
            side.cells[b].key = name[1];
            renderCells(side);
            checkHero(side);
            return '毛笔改字 · 试凑 ' + name;
          }
        }
        const slot = side.tray.indexOf(null);
        if (slot < 0) return '托盘已满，无法落笔';
        const names = Object.keys(HEROES);
        const name = names[Math.floor(Math.random() * names.length)];
        side.tray[slot] = { type: 'char', key: Math.random() < 0.5 ? name[0] : name[1] };
        renderTray(side);
        return '毛笔落纸 · 获得武将单字';
      }
    },
    {
      id: 'nongmin', name: '农民', icon: '农', price: 4, desc: '20 秒馒头产出翻倍',
      apply: function (side) { side.incomeUntil = state.time + 20; side.incomeMult = 2; return '农民助阵 · 产出翻倍'; }
    },
    {
      id: 'gongsu', name: '攻速符', icon: '速', price: 4, desc: '12 秒全军攻速 +43%',
      apply: function (side) { side.rateBuffUntil = state.time + 12; side.rateBuff = 0.7; return '攻速符 · 出手更快'; }
    },
    {
      id: 'chutu', name: '洛阳铲', icon: '铲', price: 3, desc: '获得一把铲子清理石堆',
      apply: function (side) {
        const slot = side.tray.indexOf(null);
        if (slot < 0) return '托盘已满';
        side.tray[slot] = { type: 'shovel', key: 'shovel' };
        renderTray(side);
        return '获得铲子一把';
      }
    },
    {
      id: 'huixue', name: '灵丹', icon: '丹', price: 8, desc: '阿斗恢复 1 点生命',
      apply: function (side) {
        if (side.hp >= MAX_HP) return '阿斗生命已满';
        side.hp++;
        return '灵丹生效 · 阿斗回血';
      }
    }
  ];

  function clearDrag() {
    drag = null;
    selected = null;
    tapCell = null;
    el.ghost.classList.remove('show', 'gold');
    for (const c of state.player.cellEls) c.classList.remove('drop-ok', 'drop-bad');
    for (const s of el.tray.children) s.classList.remove('selected');
  }

  function openMerchant() {
    if (!state.running || state.merchantOpen || state.userPaused) return;
    clearDrag();
    clearInspect();
    const pool = MERCHANT_ITEMS.slice();
    state.merchantChoices = [];
    for (let i = 0; i < 3; i++) {
      state.merchantChoices.push({ item: pool.splice(Math.floor(Math.random() * pool.length), 1)[0], sold: false });
    }
    state.merchantOpen = true;
    state.merchantTime = 8;
    renderMerchant();
    el.merchant.classList.remove('hidden');
    updateHud();
  }

  function renderMerchant() {
    el.merchantGold.textContent = String(state.player.gold);
    el.merchantCards.innerHTML = '';
    state.merchantChoices.forEach((choice, i) => {
      const item = choice.item;
      const card = document.createElement('div');
      card.className = 'merchant-card' + (choice.sold ? ' sold' : '') + (state.player.gold < item.price ? ' cant' : '');
      card.dataset.index = String(i);
      card.innerHTML = '<div class="m-icon">' + item.icon + '</div><div class="m-name">' + item.name + '</div>' +
        '<div class="m-desc">' + item.desc + '</div><div class="m-price">' +
        (choice.sold ? '已购' : item.price + ' 金币') + '</div>';
      el.merchantCards.appendChild(card);
    });
  }

  function closeMerchant() {
    el.merchant.classList.add('hidden');
    state.merchantOpen = false;
    state.merchantChoices = [];
    updateHud();
  }

  function updateMerchant(dt) {
    state.merchantTime -= dt;
    el.merchantTimer.textContent = Math.max(0, Math.ceil(state.merchantTime)) + 's';
    if (state.merchantTime <= 0) closeMerchant();
  }

  function togglePause() {
    if (!state.running || state.merchantOpen) return;
    clearDrag();
    state.userPaused = !state.userPaused;
    el.pausePanel.classList.toggle('hidden', !state.userPaused);
    el.pauseBtn.textContent = state.userPaused ? '继续' : '暂停';
    updateHud();
  }

  function showArmory(back) {
    let html = '<h2>军 械 库</h2><p class="sub">专属武器只对同名武将生效，品质越高加伤越多</p><div class="armory-list">';
    for (const hero of Object.keys(WEAPON_NAMES)) {
      const equipped = save.weapons.find((w) => w.id === save.equipped[hero]);
      html += '<div class="armory-row"><span class="armory-hero">' + hero + '</span>' +
        '<span class="armory-equipped">当前：' + (equipped ? weaponName(equipped) : '未装备') + '</span><div class="weapon-chips">';
      const owned = save.weapons.filter((w) => w.hero === hero).sort((a, b) => a.q - b.q);
      if (!owned.length) html += '<span class="armory-equipped">暂无专属武器</span>';
      for (const w of owned) {
        html += '<span class="weapon-chip' + (save.equipped[hero] === w.id ? ' active' : '') +
          '" data-hero="' + hero + '" data-id="' + w.id + '">' + weaponName(w) + '</span>';
      }
      if (equipped) html += '<span class="weapon-chip" data-hero="' + hero + '" data-id="0">卸下</span>';
      html += '</div></div>';
    }
    html += '</div><button id="armoryBack">返 回</button>';
    el.panel.innerHTML = html;
    el.overlay.classList.remove('hidden');
    $('armoryBack').addEventListener('click', () => {
      if (back === 'intro') showIntro();
      else if (lastResult) showResult(lastResult.win, lastResult.sec, lastResult.drop);
    });
    el.panel.querySelectorAll('.weapon-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const hero = chip.dataset.hero;
        const id = Number(chip.dataset.id);
        if (id === 0) delete save.equipped[hero];
        else save.equipped[hero] = id;
        persist();
        showArmory(back);
      });
    });
  }

  let lastResult = null;
  function showResult(win, sec, drop) {
    el.panel.innerHTML =
      '<h2>' + (win ? '大 捷' : '败 走') + '</h2>' +
      '<p class="sub">' + (win ? '敌军阿斗已失，汉室得保' : '我军阿斗失守，来日再战') + '</p>' +
      '<div class="result-line">坚持到第 <b>' + state.wave + '</b> 波 · 用时 <b>' +
      String(Math.floor(sec / 60)).padStart(2, '0') + ':' + String(sec % 60).padStart(2, '0') + '</b></div>' +
      '<div class="drop-line">获得武器：' + weaponName(drop) + '</div>' +
      '<button id="againBtn">再 战 一 局</button>' +
      '<button class="back" id="resultArmory">军 械 库</button>' +
      '<div class="record">最高记录：第 ' + save.best + ' 波</div>';
    el.overlay.classList.remove('hidden');
    $('againBtn').addEventListener('click', reset);
    $('resultArmory').addEventListener('click', () => showArmory('result'));
  }

  function finish(win) {
    state.running = false;
    const sec = Math.floor(state.time);
    if (state.wave > save.best) { save.best = state.wave; persist(); }
    const drop = dropWeapon(win);
    lastResult = { win: win, sec: sec, drop: drop };
    clearInspect();
    closeMerchant();
    state.userPaused = false;
    el.pausePanel.classList.add('hidden');
    el.pauseBtn.textContent = '暂停';
    showResult(win, sec, drop);
  }

  function checkEnd() {
    if (!state.running) return;
    if (state.player.hp <= 0) finish(false);
    else if (state.ai.hp <= 0) finish(true);
  }

  function reset() {
    el.overlay.classList.add('hidden');
    closeMerchant();
    state.userPaused = false;
    el.pausePanel.classList.add('hidden');
    el.pauseBtn.textContent = '暂停';
    clearDrag();
    clearInspect();
    state.time = 0;
    state.wave = 0;
    state.waveTimer = 2.5;
    state.projectiles.forEach((p) => p.node.remove());
    state.projectiles = [];
    const path = generatePath();
    for (const side of [state.player, state.ai]) {
      newBoard(side, path);
      side.mantou = 18;
      side.gold = 0;
      side.dmgBuffUntil = 0; side.dmgBuff = 1;
      side.rateBuffUntil = 0; side.rateBuff = 1;
      side.incomeUntil = 0; side.incomeMult = 1;
      side.hp = MAX_HP;
      side.cells.fill(null);
      side.tray.fill(null);
      side.enemies.forEach((e) => e.node.remove());
      side.enemies = [];
      side.spawnQueue = [];
      side.incomeTimer = 0;
      side.aiRecruit = 1.5; side.aiPlace = 0.5; side.aiSynth = 1.5;
      side.hasZhao = false;
      side.cellLayer.querySelectorAll('.cell').forEach((c) => { c.dataset.sig = 'reset'; });
      side.fxLayer.innerHTML = '';
    }
    el.tray.dataset.sig = '';
    el.buffs.dataset.sig = '';
    renderCells(state.player);
    renderCells(state.ai);
    renderTray(state.player);
    updateHud();
    state.running = true;
    state.lastFrame = performance.now();
  }

  function showIntro() {
    el.panel.innerHTML =
      '<h2>三英戏吕布</h2>' +
      '<p class="sub">文字合成 · 塔防对战 · 单局三到五分钟</p>' +
      '<div class="rules">' +
      '1. 点 <b>征兵</b> 花 3 馒头抽字牌，拖到棋盘布阵。<br>' +
      '2. 两个<b>同种同级</b>兵种叠在一起即可升级。<br>' +
      '3. 金色是武将<b>单字</b>，凑齐姓名横向相邻摆放即可唤将，武将<b>占两格</b>。<br>' +
      '4. <b>铲子</b>清石堆；虚线是敌军通道，通道格不能布阵。<br>' +
      '5. 点<b>商人</b>用金币买道具，点<b>暂停</b>可随时停局。<br>' +
      '6. 点<b>武将</b>看攻击范围、攻速与伤害：<b>刀</b>近战、<b>枪</b>中程、<b>弓</b>远程。<br>' +
      '7. 阿斗 <b>3 点血</b>，先被打光的一方落败。' +
      '</div>' +
      '<button id="startBtn">开 始 对 局</button>' +
      '<button class="back" id="introArmory">军 械 库</button>';
    el.overlay.classList.remove('hidden');
    $('startBtn').addEventListener('click', reset);
    $('introArmory').addEventListener('click', () => showArmory('intro'));
  }

  function frame(ts) {
    const dt = Math.min(0.05, (ts - state.lastFrame) / 1000 || 0);
    state.lastFrame = ts;
    if (state.running) {
      if (isFrozen()) {
        if (state.merchantOpen) updateMerchant(dt);
        requestAnimationFrame(frame);
        return;
      }
      state.time += dt;
      state.waveTimer -= dt;
      if (state.waveTimer <= 0) {
        state.waveTimer += WAVE_INTERVAL;
        startWave();
      }
      updateSide(state.player, dt);
      updateSide(state.ai, dt);
      updateProjectiles(dt);
      renderCells(state.player);
      renderCells(state.ai);
      checkEnd();
      state.hudTimer -= dt;
      if (state.hudTimer <= 0) {
        state.hudTimer = 0.12;
        updateHud();
      }
      if (state.hintTimer > 0) {
        state.hintTimer -= dt;
        if (state.hintTimer <= 0) el.hint.textContent = '拖动字牌入阵 · 同类合成升级 · 凑齐姓名唤武将';
      }
    }
    requestAnimationFrame(frame);
  }

  let drag = null;
  let selected = null;
  let tapCell = null;

  function setHighlight(idx) {
    for (const c of state.player.cellEls) c.classList.remove('drop-ok', 'drop-bad');
    if (idx === null || idx === undefined) return;
    const c = state.player.cellEls[idx];
    if (!c) return;
    const item = drag && drag.source === 'tray' ? state.player.tray[drag.index] : null;
    const ok = item
      ? (item.type === 'shovel' ? state.player.blocked.has(idx) :
          (!state.player.blocked.has(idx) && !state.player.pathCells.has(idx) && !state.player.cells[idx]))
      : true;
    c.classList.add(ok ? 'drop-ok' : 'drop-bad');
  }

  function cellIndexAt(x, y) {
    const node = document.elementFromPoint(x, y);
    if (!node) return null;
    const cell = node.closest && node.closest('.cell');
    if (!cell || !cell.closest('#playerGrid')) return null;
    return Number(cell.dataset.index);
  }

  function applyDrop(source, index, targetIdx) {
    const side = state.player;
    if (!state.running || isFrozen() || targetIdx === null || targetIdx === undefined) return false;
    if (side.pathCells.has(targetIdx) &&
        !(source === 'tray' && side.tray[index] && side.tray[index].type === 'shovel')) {
      showHint('虚线通道格不能布阵');
      return false;
    }
    if (source === 'tray') {
      const item = side.tray[index];
      if (!item) return false;
      if (item.type === 'shovel') {
        if (!side.blocked.has(targetIdx)) { showHint('铲子只能清理石堆格'); return false; }
        side.blocked.delete(targetIdx);
        side.tray[index] = null;
        showHint('石堆已清理，腾出一格');
      } else {
        if (side.cells[targetIdx] || side.blocked.has(targetIdx)) { showHint('此格已被占用'); return false; }
        placeAt(side, targetIdx, item);
        side.tray[index] = null;
      }
      renderTray(side);
      renderCells(side);
      return true;
    }
    if (source === 'cell') {
      if (index === targetIdx) return false;
      const a = side.cells[index];
      const b = side.cells[targetIdx];
      if (!a) return false;
      if (a.span2) { showHint('武将占地两格，暂不可移动'); return false; }
      if (side.blocked.has(targetIdx)) { showHint('石堆格需要先清理'); return false; }
      if (b && a.kind === 'base' && b.kind === 'base' && a.key === b.key && a.lv === b.lv && a.lv < MAX_LV) {
        side.cells[targetIdx] = { kind: 'base', key: a.key, lv: a.lv + 1, atk: 0, shots: 0 };
        side.cells[index] = null;
        showHint(a.key + ' 升至 ' + (a.lv + 1) + ' 级');
      } else if (!b) {
        side.cells[targetIdx] = a;
        side.cells[index] = null;
        checkHero(side);
      } else {
        showHint('只有同种同级的兵种才能合成');
        return false;
      }
      renderCells(side);
      return true;
    }
    return false;
  }

  document.addEventListener('pointerdown', (e) => {
    if (!state.running || isFrozen()) return;
    const slot = e.target.closest && e.target.closest('.slot.filled');
    if (slot) {
      if (state.inspect) clearInspect();
      const idx = Number(slot.dataset.index);
      drag = { source: 'tray', index: idx, x: e.clientX, y: e.clientY, moved: false };
      el.ghost.textContent = itemLabel(state.player.tray[idx]);
      el.ghost.classList.toggle('gold', state.player.tray[idx].type === 'char');
      el.ghost.style.left = e.clientX + 'px';
      el.ghost.style.top = e.clientY + 'px';
      return;
    }
    const unitNode = e.target.closest && e.target.closest('.unit');
    if (unitNode) {
      const cell = unitNode.closest('.cell');
      if (cell && cell.closest('#playerGrid')) {
        const idx = Number(cell.dataset.index);
        drag = { source: 'cell', index: idx, x: e.clientX, y: e.clientY, moved: false };
        el.ghost.textContent = state.player.cells[idx].key;
        el.ghost.classList.toggle('gold', state.player.cells[idx].kind !== 'base');
        el.ghost.style.left = e.clientX + 'px';
        el.ghost.style.top = e.clientY + 'px';
        return;
      }
    }
    const cell = e.target.closest && e.target.closest('.cell');
    if (cell && cell.closest('#playerGrid')) tapCell = Number(cell.dataset.index);
  });

  document.addEventListener('pointermove', (e) => {
    if (!drag) return;
    if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 6) {
      drag.moved = true;
      el.ghost.classList.add('show');
      el.ghost.style.left = e.clientX + 'px';
      el.ghost.style.top = e.clientY + 'px';
      setHighlight(cellIndexAt(e.clientX, e.clientY));
    }
  });

  document.addEventListener('pointerup', (e) => {
    if (!state.running || isFrozen()) { clearDrag(); return; }
    if (drag) {
      const idx = cellIndexAt(e.clientX, e.clientY);
      if (drag.moved && idx !== null) {
        if (applyDrop(drag.source, drag.index, idx)) selected = null;
      } else if (!drag.moved) {
        const u = drag.source === 'cell' ? state.player.cells[drag.index] : null;
        if (u && u.kind === 'hero') {
          // 武将不可移动，单击改为查看属性与射程。
          setInspect(drag.index);
          selected = null;
        } else {
          selected = (selected && selected.source === drag.source && selected.index === drag.index)
            ? null : { source: drag.source, index: drag.index };
          showHint(selected ? '已选中，点目标格放置或合成' : '已取消选择');
        }
      }
      drag = null;
      el.ghost.classList.remove('show', 'gold');
      setHighlight(null);
    } else if (tapCell !== null) {
      if (state.inspect) clearInspect();
      if (selected) {
        if (applyDrop(selected.source, selected.index, tapCell)) selected = null;
      }
      tapCell = null;
    }
    for (const s of el.tray.children) s.classList.remove('selected');
    if (selected && selected.source === 'tray' && el.tray.children[selected.index]) {
      el.tray.children[selected.index].classList.add('selected');
    }
  });

  el.recruit.addEventListener('click', () => {
    if (!state.running || isFrozen()) return;
    const side = state.player;
    const slot = side.tray.indexOf(null);
    if (slot < 0 || side.mantou < RECRUIT_COST) {
      shake(slot < 0 ? el.tray : el.recruit);
      showHint(slot < 0 ? '字牌托盘已满，先布阵' : '馒头不足');
      return;
    }
    side.mantou -= RECRUIT_COST;
    side.tray[slot] = randomItem();
    renderTray(side);
    updateHud();
  });

  el.merchantBtn.addEventListener('click', openMerchant);
  el.pauseBtn.addEventListener('click', togglePause);
  el.resumeBtn.addEventListener('click', togglePause);
  $('merchantClose').addEventListener('click', closeMerchant);
  el.heroCardClose.addEventListener('click', clearInspect);

  el.merchantCards.addEventListener('click', (e) => {
    const card = e.target.closest('.merchant-card');
    if (!card || card.classList.contains('sold')) return;
    const choice = state.merchantChoices[Number(card.dataset.index)];
    if (!choice || choice.sold) return;
    if (state.player.gold < choice.item.price) { showHint('金币不足'); return; }
    state.player.gold -= choice.item.price;
    choice.sold = true;
    showHint(choice.item.apply(state.player));
    renderMerchant();
    updateHud();
  });

  window.addEventListener('resize', () => {
    for (const side of [state.player, state.ai]) side.enemies.forEach((e) => placeEnemy(side, e));
    refreshInspectView();
  });

  if (location.search.indexOf('debug=1') >= 0) {
    window.__zyad = { state: state, save: save, placeAt: placeAt, checkHero: checkHero, renderCells: renderCells };
  }

  state.player = makeSide(false);
  state.ai = makeSide(true);
  newBoard(state.player, generatePath());
  newBoard(state.ai, state.player.pts.map((p) => ({ row: p.row, col: p.col })));
  renderCells(state.player);
  renderCells(state.ai);
  renderTray(state.player);
  showIntro();
  requestAnimationFrame(frame);
})();
