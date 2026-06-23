import { SlashCommand } from '../../slash-commands/SlashCommand.js';
import { SlashCommandParser } from '../../slash-commands/SlashCommandParser.js';
import { getContext, extension_settings, disableExtension } from '../../extensions.js';
import { addOneMessage, chat, system_avatar } from '../../../script.js';
import { getMessageTimeStamp } from '../../RossAscends-mods.js';
// eventSource, event_types, saveSettingsDebounced accessed via getContext() to avoid
// circular dependency with script.js which would break slash command registration.

console.log('[AIKP] Extension v2.0.0 loaded');

// -- Slash Commands --
// ST slash command callbacks MUST call sendSystemMessage() / toastr to display
// output. The return value is a PIPE (command chaining), NOT sent to chat.
// See docs/aicontext.md "ST Integration Code Pattern" for the rationale.

// Build a narrator-style message that preserves newlines in chat.
// GENERIC-type system messages collapse whitespace; narrator messages don't.
function sysMsg(text) {
    chat.push({
        name: 'AIKP',
        is_user: false,
        is_system: false,
        send_date: getMessageTimeStamp(),
        mes: text,
        force_avatar: system_avatar,
        extra: { type: 'aikp_command', isSmallSys: true },
    });
    addOneMessage(chat[chat.length - 1]);
    return '';
}

SlashCommandParser.addCommandObject(SlashCommand.fromProps({
    name: 'status',
    callback: async () => {
        const s = await fetchSession();
        if (!s) return sysMsg('No active session. Start a chat with a world book first.');
        const p = s.player_state || {};
        return sysMsg([
            `HP: ${p.hp || '?'}/${p.max_hp || '?'} | SAN: ${p.san || '?'}/${p.max_san || '?'}`,
            `Scene: ${p.current_scene || '?'}`,
            `Inventory: ${(p.inventory || []).join(', ') || 'empty'}`,
            `Turn: ${s.current_turn || 0} | Phase: ${s.plot_phase || 'intro'}`,
        ].join('\n'));
    },
    helpString: 'Show player status (HP, SAN, scene, inventory, turn, phase)',
}));

SlashCommandParser.addCommandObject(SlashCommand.fromProps({
    name: 'inv', aliases: ['inventory'],
    callback: async () => {
        const s = await fetchSession();
        if (!s) return sysMsg('No active session.');
        return sysMsg('Inventory: ' + ((s.player_state?.inventory || []).join(', ') || 'empty'));
    },
    helpString: 'Show inventory',
}));

SlashCommandParser.addCommandObject(SlashCommand.fromProps({
    name: 'scene',
    callback: async () => {
        const s = await fetchSession();
        if (!s) return sysMsg('No active session.');
        return sysMsg(`Scene: ${s.player_state?.current_scene || '?'} (Turn ${s.current_turn || 0}, Phase: ${s.plot_phase || 'intro'})`);
    },
    helpString: 'Show current scene',
}));

SlashCommandParser.addCommandObject(SlashCommand.fromProps({
    name: 'clues',
    callback: async () => {
        const s = await fetchSession();
        if (!s) return sysMsg('No active session.');
        const found = Object.entries(s.entity_states || {})
            .filter(([, v]) => ['found', 'read', 'opened'].includes(v))
            .map(([k]) => k);
        return sysMsg('Discovered clues: ' + (found.length ? found.join(', ') : 'none'));
    },
    helpString: 'List discovered clues',
}));

SlashCommandParser.addCommandObject(SlashCommand.fromProps({
    name: 'reset',
    callback: async () => {
        try {
            await fetch(`${BACKEND_URL}/api/session/${await getChatId()}/reset`, { method: 'POST' });
            _cachedChatId = null;
            return sysMsg('Session reset. Start a new message to begin fresh.');
        } catch (e) { return sysMsg('Error: ' + e.message); }
    },
    helpString: 'Reset game session',
}));

SlashCommandParser.addCommandObject(SlashCommand.fromProps({
    name: 'aikp-help',
    callback: () => sysMsg([
        '/status - 玩家状态（生命、理智、场景、背包）',
        '/inv - 查看背包',
        '/scene - 当前场景',
        '/clues - 已发现线索',
        '/reset - 重置会话',
        '/check <技能> [难度] - 手动骰子检定（例如 /check 察觉 12）',
        '/guide on|off - 开关 GM 提示',
    ].join('\n')),
    helpString: 'Show AIKP commands',
}));

SlashCommandParser.addCommandObject(SlashCommand.fromProps({
    name: 'check',
    callback: async (args, value) => {
        const parts = (value || '').trim().split(/\s+/);
        const skill = parts[0] || '察觉';
        const dc = parseInt(parts[1]) || 12;
        try {
            const r = await fetch(`${BACKEND_URL}/api/session/${await getChatId()}`);
            if (!r.ok) return sysMsg('No active session.');
            const s = await r.json();
            const sv = s.player_state?.skills?.[skill] || 0;
            const d20 = Math.floor(Math.random() * 20) + 1;
            const total = d20 + sv;
            const verdict = d20 === 20 ? 'CRITICAL SUCCESS!' :
                d20 === 1 ? 'CRITICAL FAILURE!' :
                total >= dc ? 'SUCCESS' : 'FAILURE';
            return sysMsg(`${skill} check: d20=${d20} + ${sv} = ${total} (DC=${dc}) -> ${verdict}`);
        } catch (e) { return sysMsg('Error: ' + e.message); }
    },
    helpString: 'Manual dice roll: /check <skill> [dc] (default: 察觉, DC=12)',
}));

SlashCommandParser.addCommandObject(SlashCommand.fromProps({
    name: 'guide',
    callback: async (args, value) => {
        const toggle = (value || '').trim().toLowerCase();
        if (toggle === 'on') {
            extension_settings.aikp = extension_settings.aikp || {};
            extension_settings.aikp.guideEnabled = true;
            getContext().saveSettingsDebounced();
            return sysMsg('GM hints enabled.');
        } else if (toggle === 'off') {
            extension_settings.aikp = extension_settings.aikp || {};
            extension_settings.aikp.guideEnabled = false;
            getContext().saveSettingsDebounced();
            return sysMsg('GM hints disabled.');
        }
        const state = extension_settings.aikp?.guideEnabled !== false ? 'enabled' : 'disabled';
        return sysMsg(`GM hints are currently ${state}. Use /guide on or /guide off to toggle.`);
    },
    helpString: 'Toggle GM hints: /guide on|off',
}));

// -- Constants -------------------------------------------------
const BACKEND_URL = 'http://localhost:8001';
let _cachedChatId = null;
let _aikpActive = false;

// -- AIKP Mode Detection --------------------------------------

function isAikpMode() {
    if (_aikpActive) return true;
    try {
        const url = $('#custom_api_url_text').val() || '';
        const source = $('#chat_completion_source').val() || '';
        _aikpActive = source === 'custom' &&
            (url.includes('localhost:8001') || url.includes('127.0.0.1:8001'));
    } catch (_) {
        _aikpActive = false;
    }
    return _aikpActive;
}

// -- Backend API helpers --------------------------------------

async function getChatId() {
    // Must match backend chat_id = "{selected_model}-session". Prefer the model
    // actually selected in ST, NOT /v1/models[0] (just the first world book
    // alphabetically) — using [0] makes the panel read the wrong/empty session.
    try {
        const sel = ($('#custom_model_id').val() || $('#model_custom_select').val() || '').trim();
        if (sel) return `${sel}-session`;
    } catch (_) {}
    if (_cachedChatId) return _cachedChatId;
    try {
        const r = await fetch(`${BACKEND_URL}/v1/models`);
        const d = await r.json();
        _cachedChatId = (d.data?.[0]?.id || 'tavern_trial') + '-session';
    } catch {
        _cachedChatId = 'tavern_trial-session';
    }
    return _cachedChatId;
}

async function fetchSession() {
    try {
        const r = await fetch(`${BACKEND_URL}/api/session/${await getChatId()}`);
        if (r.ok) return await r.json();
    } catch (e) { console.error('[AIKP] fetchSession:', e.message); }
    return null;
}

async function fetchWorlds() {
    try {
        const r = await fetch(`${BACKEND_URL}/api/worlds`);
        if (r.ok) return await r.json();
    } catch (e) { console.error('[AIKP] fetchWorlds:', e.message); }
    return [];
}

// -- Refresh ST Model List -------------------------------------

// refreshModelList triggers ST to re-fetch /v1/models and re-populate
// the model selector dropdown. Uses the existing reconnect flow.
function refreshModelList() {
    try {
        // Trigger the reconnect handler which fetches /v1/models for CUSTOM source
        $('#connect_button').trigger('click');
    } catch (e) {
        console.error('[AIKP] refreshModelList failed:', e);
    }
}

// -- Conflict Handling -----------------------------------------

function handleConflicts() {
    if (isAikpMode()) {
        // Disable ST's memory/summarize extension.
        // AIKP has built-in conversation summary every 10 turns via state_manager.py.
        // Having both active causes duplicate LLM calls and conflicting summaries.
        const disabled = extension_settings.disabledExtensions || [];
        if (!disabled.includes('memory')) {
            console.log('[AIKP] Disabling ST memory extension (AIKP provides built-in summary)');
            disableExtension('memory', false).catch(e =>
                console.warn('[AIKP] Failed to disable memory extension:', e));
        }
    } else {
        // Re-enable ST memory extension when AIKP is deactivated
        const disabled = extension_settings.disabledExtensions || [];
        if (disabled.includes('memory')) {
            console.log('[AIKP] Re-enabling ST memory extension');
            import('../../extensions.js').then(m => m.enableExtension('memory', false)).catch(e =>
                console.warn('[AIKP] Failed to re-enable memory extension:', e));
        }
    }
}

// -- Message Trimming ------------------------------------------

// ST assembles a complex multi-layer prompt (World Info, character card,
// extension prompts, conversation, etc.). AIKP's engine rebuilds context
// from session data (state snapshot + entity memories + RAG + conversation
// summary) and only needs character identity + recent dialogue from ST.
function onPromptReady(data) {
    if (!isAikpMode()) return;
    if (!data.chat || !Array.isArray(data.chat)) return;

    const slim = [];
    // Keep the first system message (character identity info)
    const firstSystem = data.chat.find(m => m.role === 'system');
    if (firstSystem) slim.push(firstSystem);
    // Keep last 15 user/assistant dialogue turns
    const dialogue = data.chat.filter(m => m.role !== 'system').slice(-15);
    slim.push(...dialogue);
    data.chat = slim;
}

// -- Top Bar State Display -------------------------------------

async function updateStatusBar() {
    if (!isAikpMode()) return;
    const session = await fetchSession();
    if (!session) return;
    const ps = session.player_state || {};
    const sceneName = ps.current_scene_name || ps.current_scene || '?';
    const name = ps.name || '调查员';
    const prof = ps.profession ? `（${ps.profession}）` : '';

    // Skills: keep only Chinese-named real skills (drop attribute/derived aliases)
    const ATTRS = ['力量', '敏捷', '意志', '体质', '外貌', '教育', '体型', '智力', '幸运'];
    const DERIVED = ['理智', '体力', '魔法'];
    const skills = ps.skills || {};
    const skillStr = Object.entries(skills)
        .filter(([k, v]) => v && /^[一-鿿①-⑩：:]+$/.test(k)
            && !ATTRS.includes(k) && !DERIVED.includes(k))
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([k, v]) => `${k}${v}`)
        .join(' ');

    const html = `
        <div class="aikp-sp-line"><strong>${name}</strong>${prof} &middot; ${sceneName} &middot; T${session.current_turn || 0} &middot; ${session.plot_phase || 'intro'}</div>
        <div class="aikp-sp-line">HP ${ps.hp ?? '?'}/${ps.max_hp ?? '?'} &nbsp; SAN ${ps.san ?? '?'}/${ps.max_san ?? '?'} &nbsp; MP ${ps.mp ?? '?'}/${ps.max_mp ?? '?'}</div>
        ${skillStr ? `<div class="aikp-sp-line aikp-sp-skills">${skillStr}</div>` : ''}
    `;
    const panel = $('#aikp_status_panel');
    if (panel.length) panel.html(html);

    // Dice button: show & highlight when a check awaits the player's roll
    const pc = session.pending_check;
    const dice = $('#aikp_dice_button');
    if (dice.length) {
        if (pc) {
            const parts = [];
            if (pc.skill) parts.push(`〈${pc.skill}〉`);
            if (pc.san_check) parts.push('理智');
            dice.attr('title', `点击掷骰：${parts.join('+')}检定`).addClass('aikp-dice-active');
        } else {
            dice.attr('title', '掷骰（现在没有待掷的检定）').removeClass('aikp-dice-active');
        }
    }
}

// Player clicks the dice → backend rolls the pending check and returns the result.
async function rollPendingCheck() {
    let chatId;
    try { chatId = await getChatId(); } catch { return; }
    try {
        const r = await fetch(`${BACKEND_URL}/api/roll/${encodeURIComponent(chatId)}`, { method: 'POST' });
        if (!r.ok) { toastr.warning('现在没有待掷的检定', 'AIKP'); return; }
        const d = await r.json();
        const lines = [];
        if (d.check) {
            lines.push(`🎲 〈${d.check.skill}〉检定：掷出 ${d.check.roll}（目标 ≤${d.check.target}）→ <b>${d.check.verdict_cn}</b>`);
        }
        if (d.san) {
            let s = `🎲 理智检定：d100=${d.san.roll} vs 当前SAN ${d.san.vs_san} → ${d.san.passed ? '成功' : '失败'}，损失 ${d.san.loss}（${d.san.before}→${d.san.after}）`;
            if (d.san.insanity_temp) s += ' ⚠临时疯狂';
            if (d.san.insanity_indef) s += ' ⚠⚠不定性疯狂';
            lines.push(s);
        }
        toastr.info(lines.join('<br>'), 'AIKP 掷骰', { timeOut: 9000, escapeHtml: false });
        if (d.narration) toastr.success(d.narration, '结果', { timeOut: 12000 });
        $('#aikp_dice_button').removeClass('aikp-dice-active');
        updateStatusBar();
    } catch (e) {
        toastr.error('掷骰失败: ' + e.message, 'AIKP');
    }
}

// -- World Book Management Panel -------------------------------

async function renderWorldPanel() {
    const worlds = await fetchWorlds();
    const listHtml = worlds.length ? worlds.map(w => `
        <div class="aikp-world-item">
            <div>
                <div><strong>${w.name || w.id}</strong></div>
                <div class="aikp-world-meta">
                    ${w.scene_count ?? '?'} scenes, ${w.entity_count ?? '?'} entities
                </div>
            </div>
            <div class="aikp-world-actions">
                <button class="aikp-btn" onclick="window._aikp_selectWorld('${w.id}')">Select</button>
                <button class="aikp-btn danger" onclick="window._aikp_deleteWorld('${w.id}')">Delete</button>
            </div>
        </div>
    `).join('') : '<div style="padding:8px;color:var(--SmartThemeEmColor)">No world books parsed yet. Upload a module to get started.</div>';

    $('#aikp_world_list').html(listHtml);
}

// Expose global functions for button onclick handlers
window._aikp_selectWorld = function(worldId) {
    try {
        $('#custom_model_id').val(worldId).trigger('input');
        $('#model_custom_select').val(worldId).trigger('change');
        _cachedChatId = null;
        toastr.success(`Switched to "${worldId}"`, 'AIKP');
    } catch (e) {
        toastr.error('Failed to switch world', 'AIKP');
    }
};

window._aikp_deleteWorld = async function(worldId) {
    if (!confirm(`Delete world book "${worldId}"? This cannot be undone.`)) return;
    try {
        const r = await fetch(`${BACKEND_URL}/api/worlds/${worldId}`, { method: 'DELETE' });
        if (r.ok) {
            toastr.success(`"${worldId}" deleted`, 'AIKP');
            refreshModelList();
            renderWorldPanel();
        } else {
            const d = await r.json().catch(() => ({}));
            toastr.error(d.detail || 'Delete failed', 'AIKP');
        }
    } catch (e) {
        toastr.error('Delete failed: ' + e.message, 'AIKP');
    }
};

// -- Module Upload ---------------------------------------------

async function uploadModule(file) {
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    if (!['txt', 'md', 'docx', 'pdf'].includes(ext)) {
        toastr.error(`Unsupported format: .${ext}. Use .txt, .md, .docx, or .pdf`, 'AIKP');
        return;
    }

    toastr.info(`Uploading "${file.name}"...`, 'AIKP', { timeOut: 2000 });
    const progressFill = $('#aikp_upload_progress_fill');
    const statusText = $('#aikp_upload_status');
    const spinner = $('#aikp_upload_spinner');
    progressFill.css('width', '5%');
    statusText.text('Reading file...');
    spinner.show();

    let text = '';
    try {
        if (ext === 'pdf') {
            // Reuse ST's client-side pdf.js for text extraction.
            // pdf.min.mjs is lazy-loaded by extractTextFromPDF in utils.js.
            const { extractTextFromPDF } = await import('../../utils.js');
            text = await extractTextFromPDF(file);
            if (!text || !text.trim()) {
                toastr.error(
                    'No extractable text found in PDF. If this is a scanned document, try converting to text first.',
                    'AIKP',
                    { timeOut: 8000 }
                );
                return;
            }
        } else if (ext === 'docx') {
            // Upload raw .docx to backend for zipfile-based text extraction
            const form = new FormData();
            form.append('file', file);
            progressFill.css('width', '10%');
            statusText.text('Uploading...');
            const uploadResp = await fetch(`${BACKEND_URL}/api/upload`, { method: 'POST', body: form });
            const uploadData = await uploadResp.json().catch(() => ({}));
            if (!uploadResp.ok) throw new Error(uploadData.detail || 'Upload failed');
            text = uploadData.preview || '';
            // Trigger parse from upload_id
            statusText.text('Starting parse...');
            const parseResp = await fetch(`${BACKEND_URL}/api/parse/${uploadData.upload_id}`, { method: 'POST' });
            const parseData = await parseResp.json().catch(() => ({}));
            if (!parseResp.ok) throw new Error(parseData.detail || 'Parse trigger failed');
            return pollParse(parseData.upload_id, file.name);
        } else {
            // .txt / .md: read as text directly
            text = await file.text();
        }
    } catch (e) {
        progressFill.css('width', '0');
        statusText.text('');
        spinner.hide();
        toastr.error('Upload failed: ' + e.message, 'AIKP');
        return;
    }

    if (!text || !text.trim()) {
        progressFill.css('width', '0');
        statusText.text('');
        spinner.hide();
        toastr.error('File is empty', 'AIKP');
        return;
    }

    // Send text to backend for parsing
    progressFill.css('width', '20%');
    statusText.text('Sending to parser...');
    try {
        const resp = await fetch(`${BACKEND_URL}/api/parse/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, filename: file.name }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Parse request failed');
        return pollParse(data.upload_id, file.name);
    } catch (e) {
        progressFill.css('width', '0');
        statusText.text('');
        spinner.hide();
        toastr.error('Parse request failed: ' + e.message, 'AIKP');
    }
}

async function pollParse(uploadId, filename) {
    const progressFill = $('#aikp_upload_progress_fill');
    const statusText = $('#aikp_upload_status');
    const spinner = $('#aikp_upload_spinner');
    let attempts = 0;
    const maxAttempts = 300; // 10 minutes at 2s intervals

    return new Promise((resolve) => {
        const poll = setInterval(async () => {
            try {
                const r = await fetch(`${BACKEND_URL}/api/parse/${uploadId}/status`);
                const d = await r.json();
                const pct = Math.min(d.progress || 0, 100);
                progressFill.css('width', pct + '%');
                statusText.text(d.step ? `${d.step} (${pct}%)` : `Parsing... (${pct}%)`);

                if (d.status === 'done') {
                    clearInterval(poll);
                    progressFill.css('width', '100%');
                    statusText.text('');
                    spinner.hide();
                    toastr.success(
                        `"${d.world_name}" ready! (${d.scene_count} scenes, ${d.entity_count} entities)`,
                        'AIKP',
                        { timeOut: 10000 }
                    );
                    // Auto-switch to new world book
                    try {
                        $('#custom_model_id').val(d.world_name).trigger('input');
                        $('#model_custom_select').val(d.world_name).trigger('change');
                        _cachedChatId = null;
                    } catch (_) {}
                    refreshModelList();
                    renderWorldPanel();
                    resolve();
                } else if (d.status === 'error') {
                    clearInterval(poll);
                    progressFill.css('width', '0');
                    statusText.text('');
                    spinner.hide();
                    toastr.error(`Parse failed: ${d.error || 'Unknown error'}`, 'AIKP', { timeOut: 15000 });
                    resolve();
                }
            } catch (e) {
                clearInterval(poll);
                progressFill.css('width', '0');
                statusText.text('');
                spinner.hide();
                console.error('[AIKP] Poll error:', e);
                toastr.error('Lost connection to backend during parse', 'AIKP');
                resolve();
            }
            if (++attempts >= maxAttempts) {
                clearInterval(poll);
                progressFill.css('width', '0');
                statusText.text('');
                spinner.hide();
                toastr.warning('Parse is taking longer than expected. Check /parse page for status.', 'AIKP');
                resolve();
            }
        }, 2000);
    });
}

// -- UI Rendering ----------------------------------------------

function renderUploadUI() {
    // File input must be at body level (not inside hidden drawers).
    // Browsers may block file picker dialogs from inputs inside display:none containers.
    const fileInputEl = document.createElement('input');
    fileInputEl.type = 'file';
    fileInputEl.id = 'aikp_file_input';
    fileInputEl.accept = '.txt,.md,.docx,.pdf';
    fileInputEl.style.display = 'none';
    document.body.appendChild(fileInputEl);
    const $fileInput = $(fileInputEl);

    // Wand menu upload button - with retry for async template loading
    function appendWandButton() {
        const container = document.getElementById('aikp_wand_container');
        if (!container) {
            console.debug('[AIKP] wand container not ready, retrying in 500ms...');
            setTimeout(appendWandButton, 500);
            return;
        }
        const wandBtn = $(`
            <div id="aikp_wand_upload" class="interactable" title="Upload TRPG Module">
                <i class="fa-solid fa-file-import"></i> Upload Module
            </div>
        `);
        // Use native click() for trusted user gesture propagation to file input
        wandBtn.on('click', () => fileInputEl.click());
        $(container).append(wandBtn);
        console.log('[AIKP] Wand menu button appended');
    }
    appendWandButton();

    // Management panel in extensions container
    const panel = $(`
        <div class="aikp-panel">
            <h4>AIKP World Books</h4>
            <div class="aikp-upload-zone" id="aikp_upload_zone">
                <p>Click or drag .txt / .md / .docx / .pdf here</p>
                <p style="color:var(--SmartThemeEmColor);font-size:0.8em">Max 50MB</p>
            </div>
            <div class="aikp-upload-hint">
                <b>📋 上传什么模组效果最好</b>
                <ul>
                    <li><b>CoC7（克苏鲁）</b>模组支持最完整：骰子检定、SAN、判定点</li>
                    <li><b>结构清晰</b>：场景、NPC、对话、判定（〈侦查〉判定 / SANcheck）写明白的</li>
                    <li><b>中文 · 纯文字</b>的 txt / docx / pdf（扫描成图片的 PDF 读不出文字）</li>
                    <li><b>篇幅适中</b>（几千～几万字）；超大模组解析慢且费 token</li>
                </ul>
                <span class="aikp-upload-hint-foot">解析后自动提取：场景路线、NPC（性格/对话/特征）、判定点（技能检定 / SAN / 成败分支）。判定时右上角 🎲 会亮，玩家点它掷骰。</span>
            </div>
            <div class="aikp-progress" id="aikp_upload_progress">
                <div class="aikp-progress-fill" id="aikp_upload_progress_fill"></div>
            </div>
            <div style="display:flex;align-items:center;gap:6px;margin-top:4px">
                <div class="aikp-spinner" id="aikp_upload_spinner" style="display:none"></div>
                <div id="aikp_upload_status" style="font-size:0.8em;color:var(--SmartThemeEmColor)"></div>
            </div>
            <div class="aikp-world-list" id="aikp_world_list">
                <div style="padding:8px;color:var(--SmartThemeEmColor)">Loading...</div>
            </div>
            <div style="margin-top:8px">
                <button class="aikp-btn" id="aikp_refresh_btn">Refresh</button>
            </div>
        </div>
    `);
    $('#aikp_container').append(panel);

    // File input change handler
    $fileInput.on('change', () => {
        if (fileInputEl.files.length) {
            uploadModule(fileInputEl.files[0]);
            $fileInput.val(''); // Reset so same file can be re-selected
        }
    });

    // Upload zone handlers - use native click for file input
    const uploadZone = $('#aikp_upload_zone');
    uploadZone.on('click', () => fileInputEl.click());
    uploadZone.on('dragover', (e) => { e.preventDefault(); uploadZone.css('border-color', '#e94560'); });
    uploadZone.on('dragleave', () => uploadZone.css('border-color', ''));
    uploadZone.on('drop', (e) => {
        e.preventDefault();
        uploadZone.css('border-color', '');
        const files = e.originalEvent.dataTransfer.files;
        if (files.length) uploadModule(files[0]);
    });

    $('#aikp_refresh_btn').on('click', () => {
        renderWorldPanel();
        refreshModelList();
    });

    // Top bar: character-status toggle button + collapsible panel
    const statusToggle = $('<div id="aikp_status_toggle" title="角色状态">🎭 角色状态 ▼</div>');
    const statusPanel = $('<div id="aikp_status_panel" style="display:none"></div>');
    // Attach to body, not #top-bar — the top bar can be hidden in some layouts,
    // which would hide the (fixed-positioned) button with it.
    $(document.body).append(statusToggle);
    $(document.body).append(statusPanel);

    // Dice button — player clicks it to roll a pending check
    const diceButton = $('<div id="aikp_dice_button" title="掷骰（轮到检定时会高亮）">🎲</div>');
    $(document.body).append(diceButton);
    diceButton.on('click', rollPendingCheck);
    statusToggle.on('click', async () => {
        if (statusPanel.is(':visible')) {
            statusPanel.hide();
            statusToggle.html('🎭 角色状态 ▼');
        } else {
            await updateStatusBar();
            statusPanel.show();
            statusToggle.html('🎭 角色状态 ▲');
        }
    });

    // Initial load
    renderWorldPanel();
}

// -- Extension Init --------------------------------------------

export function init() {
    console.log('[AIKP] init() - AIKP GM Engine v2.0.0');

    // Initialize settings
    extension_settings.aikp = extension_settings.aikp || {};
    if (extension_settings.aikp.guideEnabled === undefined) {
        extension_settings.aikp.guideEnabled = true;
    }

    // Handle conflicting extensions
    handleConflicts();

    // Register event listeners
    const ctx = getContext();
    ctx.eventSource.on(ctx.eventTypes.CHAT_COMPLETION_PROMPT_READY, onPromptReady);
    ctx.eventSource.on(ctx.eventTypes.MESSAGE_RECEIVED, updateStatusBar);
    ctx.eventSource.on(ctx.eventTypes.CHATCOMPLETION_SOURCE_CHANGED, () => {
        _aikpActive = false; // Re-evaluate on next isAikpMode() call
        handleConflicts();
    });
    ctx.eventSource.on(ctx.eventTypes.CHAT_CHANGED, () => {
        _cachedChatId = null;
        updateStatusBar();
    });

    // Render UI
    renderUploadUI();

    // Remove old Data Bank monkey-patch if it exists.
    // Data Bank is preserved as-is for future multiplayer file sharing.
    // The old Array.push proxy in aikp-upload is no longer used.
    // Module upload goes through the native AIKP panel instead.
    try {
        const ctx = getContext();
        const attachments = ctx.extensionSettings?.attachments;
        if (attachments && attachments.push !== Array.prototype.push) {
            console.log('[AIKP] Removing old Data Bank monkey-patch');
            attachments.push = Array.prototype.push;
        }
    } catch (_) {}

    console.log('[AIKP] Ready');
}
