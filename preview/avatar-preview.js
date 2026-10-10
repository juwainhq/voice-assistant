/* Shared browser renderer for the desktop-avatar previews. */
(() => {
  'use strict';

  const VALID_MODES = new Set(['portrait', 'grokbot', 'image']);
  const VALID_STATES = new Set(['idle', 'happy', 'listening', 'thinking', 'talking', 'sad']);
  let instanceId = 0;

  class NovaAvatar extends HTMLElement {
    static get observedAttributes() { return ['mode', 'state', 'size']; }

    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this._time = 0;
      this._lastTimestamp = 0;
      this._frame = 0;
      this._blinkIn = 2.7 + Math.random() * 2.2;
      this._blinkLeft = 0;
      this._mountedMode = '';
      this._request = 0;
      this._renderFrame = this._renderFrame.bind(this);
    }

    connectedCallback() {
      this._mount();
      this._request = requestAnimationFrame(this._renderFrame);
    }

    disconnectedCallback() {
      cancelAnimationFrame(this._request);
    }

    attributeChangedCallback(name, oldValue, newValue) {
      if (oldValue === newValue || !this.isConnected) return;
      if (name === 'mode' && this._mountedMode !== this._mode()) this._mount();
      if (name === 'size') this._setSize();
      if (name === 'state') this._applyAriaLabel();
    }

    _mode() {
      const mode = (this.getAttribute('mode') || 'portrait').toLowerCase();
      return VALID_MODES.has(mode) ? mode : 'portrait';
    }

    _state() {
      const state = (this.getAttribute('state') || 'idle').toLowerCase();
      return VALID_STATES.has(state) ? state : 'idle';
    }

    _setSize() {
      const value = Number.parseInt(this.getAttribute('size') || '72', 10);
      const size = Number.isFinite(value) ? Math.max(32, Math.min(value, 512)) : 72;
      this.style.width = `${size}px`;
      this.style.height = `${size}px`;
      const art = this.shadowRoot.querySelector('.avatar-art, .still-art');
      if (art) {
        const artSize = this._mode() === 'grokbot' && size <= 100
          ? Math.min(size, 48)
          : this._mode() === 'image' && size <= 100
            ? Math.min(size, 64)
            : size;
        art.style.width = `${artSize}px`;
        art.style.height = `${artSize}px`;
      }
    }

    _applyAriaLabel() {
      const modeName = this._mode() === 'portrait'
        ? 'Animated reference portrait'
        : this._mode() === 'grokbot' ? 'Classic Grok Bot' : 'Static portrait image';
      this.setAttribute('aria-label', `${modeName}, ${this._state()} state`);
    }

    _mount() {
      const mode = this._mode();
      if (!this.shadowRoot) return;
      this._mountedMode = mode;
      this.setAttribute('role', 'img');
      this._applyAriaLabel();

      const styles = `
        <style>
          :host { display: block; position: relative; flex: 0 0 auto; overflow: hidden; }
          .stage { position: absolute; inset: 0; overflow: hidden; }
          .avatar-art, .still-art {
            position: absolute; left: 50%; top: 50%; display: block;
            transform: translate(-50%, -50%);
          }
          .portrait-art { background: #201E21; }
          .grok-art { overflow: visible; }
          .still-art { object-fit: cover; }
          .status-indicator {
            position: absolute; right: 0; bottom: 0; width: 22%; height: 22%;
            overflow: visible; pointer-events: none;
          }
          .status-indicator .wave, .status-indicator .dots { display: none; }
          .status-indicator[data-state="listening"] .wave,
          .status-indicator[data-state="talking"] .wave { display: block; }
          .status-indicator[data-state="thinking"] .dots { display: block; }
        </style>`;

      if (mode === 'portrait') {
        this.shadowRoot.innerHTML = styles + this._portraitMarkup();
      } else if (mode === 'grokbot') {
        this.shadowRoot.innerHTML = styles + this._grokMarkup();
      } else {
        this.shadowRoot.innerHTML = styles + this._imageMarkup();
        const image = this.shadowRoot.querySelector('.still-art');
        image.addEventListener('error', () => {
          if (this._mode() !== 'image') return;
          this.setAttribute('mode', 'grokbot');
          this.dispatchEvent(new CustomEvent('avatarfallback', {
            bubbles: true,
            detail: { message: 'Static avatar artwork could not be loaded; showing Classic Grok Bot.' },
          }));
        }, { once: true });
      }
      this._setSize();
      this._applyAriaLabel();
    }

    _portraitMarkup() {
      return `
        <div class="stage">
          <svg class="avatar-art portrait-art" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <rect width="100" height="100" fill="#201E21"/>
            <g class="back-hair">
              <ellipse cx="49.5" cy="45.5" rx="61.5" ry="65.5" fill="#402E2A"/>
              <path d="M66 14 C83 17 91 31 90 47 C89 64 79 81 75 101 C67 103 60 98 56 93 C65 77 72 61 75 48 C77 34 72 22 66 14Z" fill="#302321"/>
            </g>
            <g class="body">
              <path d="M12 103 C15 93 23 88 31 88 C38 88 43 94 49 91 C60 86 73 91 82 103 C61 108 32 108 12 103Z" fill="#2A282C"/>
              <path d="M31 74 C37 78 49 80 57 75 C55 86 57 96 63 103 C48 105 32 104 23 103 C30 93 31 83 31 74Z" fill="#F5D2C7"/>
            </g>
            <g class="portrait-head">
              <ellipse cx="72" cy="65" rx="9" ry="10.5" transform="rotate(-7 72 65)" fill="#F5D2C7"/>
              <ellipse cx="74" cy="65" rx="3.1" ry="5" transform="rotate(-7 74 65)" fill="#EAB8AE"/>
              <path d="M39 26 C53 23 66 30 71 42 C76 54 72 68 64 78 C57 87 48 91 38 90 C25 89 15 81 10 71 C5 60 7 47 13 38 C20 29 30 25 39 26Z" fill="#FDEBE1"/>
              <path d="M-4 38 C7 40 14 48 15 59 C17 74 11 88 15 103 C8 104 1 102 -4 99 C-2 78 -2 55 -4 38Z" fill="#302321"/>
              <path d="M-4 43 C0 26 6 12 19 4 C32 -4 49 -3 61 4 C73 10 80 22 80 34 C81 41 78 47 74 50 C68 42 62 37 55 34 C47 30 39 31 32 35 C24 40 20 47 16 55 C11 63 4 68 -4 68Z" fill="#432F2B"/>
              <path d="M-4 40 C7 32 16 25 27 23 C40 19 53 21 63 28 C51 25 40 27 31 33 C23 38 18 45 14 53 C10 59 4 63 -4 63Z" fill="#302321"/>
              <path class="brow brow-left" d="M15 42 C21 39 28 40 34 44" fill="none" stroke="#3A2826" stroke-width="3.3" stroke-linecap="round"/>
              <path class="brow brow-right" d="M52 49 C59 50 66 53 71 57" fill="none" stroke="#3A2826" stroke-width="3.3" stroke-linecap="round"/>
              <ellipse class="blush blush-left" cx="15.5" cy="63" rx="6" ry="3.7" transform="rotate(14 15.5 63)" fill="#F7C8C6"/>
              <ellipse class="blush blush-right" cx="62.5" cy="72" rx="6.5" ry="3.7" transform="rotate(11 62.5 72)" fill="#F7C8C6"/>
              <g class="eye-gaze">
                <g class="open-eyes">
                  <ellipse class="portrait-eye" cx="24" cy="54.5" rx="5.35" ry="9.5" transform="rotate(-9 24 54.5)" fill="#241A1A"/>
                  <ellipse cx="22.65" cy="51.4" rx="1.05" ry="2.65" transform="rotate(-9 24 54.5)" fill="#FFF9F3"/>
                  <ellipse class="portrait-eye" cx="58" cy="64" rx="5.35" ry="9.5" transform="rotate(9 58 64)" fill="#241A1A"/>
                  <ellipse cx="56.65" cy="60.9" rx="1.05" ry="2.65" transform="rotate(9 58 64)" fill="#FFF9F3"/>
                </g>
                <g class="closed-eyes" fill="none" stroke="#241A1A" stroke-width="2.1" stroke-linecap="round" style="display:none">
                  <path d="M18 54.5 Q24 55.7 30 54.5"/>
                  <path d="M52 64 Q58 65.2 64 64"/>
                </g>
              </g>
            </g>
          </svg>
          <svg class="status-indicator" viewBox="0 0 24 24" aria-hidden="true">
            <g class="wave" fill="#FAFAFA">
              <rect class="bar" x="1" y="9" width="3" height="6" rx="1.5"/>
              <rect class="bar" x="5.2" y="6" width="3" height="12" rx="1.5"/>
              <rect class="bar" x="9.4" y="8" width="3" height="8" rx="1.5"/>
              <rect class="bar" x="13.6" y="5" width="3" height="14" rx="1.5"/>
              <rect class="bar" x="17.8" y="8" width="3" height="8" rx="1.5"/>
            </g>
            <g class="dots" fill="#FAFAFA">
              <circle cx="5" cy="12" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="19" cy="12" r="1.5"/>
            </g>
          </svg>
        </div>`;
    }

    _grokMarkup() {
      return `
        <div class="stage">
          <svg class="avatar-art grok-art" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <circle cx="53" cy="53" r="42" fill="#8C8C8C" opacity=".35"/>
            <circle cx="50" cy="50" r="42" fill="#FAFAFA"/>
            <ellipse class="classic-blush" cx="25.6" cy="67.6" rx="5.2" ry="3.6" fill="#8C8C8C" opacity=".5"/>
            <ellipse class="classic-blush" cx="74.4" cy="67.6" rx="5.2" ry="3.6" fill="#8C8C8C" opacity=".5"/>
            <ellipse class="classic-eye" cx="34.5" cy="52" rx="4.3" ry="8.6" fill="#000"/>
            <ellipse class="classic-eye" cx="65.5" cy="52" rx="4.3" ry="8.6" fill="#000"/>
          </svg>
        </div>`;
    }

    _imageMarkup() {
      return `
        <div class="stage">
          <img class="still-art" src="../avatar.png" alt="" />
          <svg class="status-indicator" viewBox="0 0 24 24" aria-hidden="true">
            <g class="wave" fill="#FAFAFA"><rect class="bar" x="1" y="9" width="3" height="6" rx="1.5"/><rect class="bar" x="5.2" y="6" width="3" height="12" rx="1.5"/><rect class="bar" x="9.4" y="8" width="3" height="8" rx="1.5"/><rect class="bar" x="13.6" y="5" width="3" height="14" rx="1.5"/><rect class="bar" x="17.8" y="8" width="3" height="8" rx="1.5"/></g>
            <g class="dots" fill="#FAFAFA"><circle cx="5" cy="12" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="19" cy="12" r="1.5"/></g>
          </svg>
        </div>`;
    }

    _renderFrame(timestamp) {
      const dt = this._lastTimestamp ? Math.min((timestamp - this._lastTimestamp) / 1000, 0.1) : 0.08;
      this._lastTimestamp = timestamp;
      this._time += dt;
      this._frame += 1;
      if (this._blinkLeft > 0) this._blinkLeft = Math.max(0, this._blinkLeft - dt);
      else {
        this._blinkIn -= dt;
        if (this._blinkIn <= 0) {
          this._blinkLeft = 0.16;
          this._blinkIn = 2.7 + Math.random() * 2.5;
        }
      }
      this._animate();
      this._request = requestAnimationFrame(this._renderFrame);
    }

    _animate() {
      const state = this._state();
      const mode = this._mode();
      const indicator = this.shadowRoot.querySelector('.status-indicator');
      if (indicator) {
        const indicatorState = mode === 'image' && state === 'talking' ? 'thinking' : state;
        indicator.dataset.state = indicatorState;
        indicator.querySelectorAll('.bar').forEach((bar, index) => {
          const pattern = [0.35, 0.85, 0.5, 1, 0.45];
          const frame = Math.floor(this._time * 8) % 4;
          const rotated = pattern[(index + frame) % pattern.length];
          const height = 4 + rotated * 13;
          bar.setAttribute('y', String(12 - height / 2));
          bar.setAttribute('height', String(height));
        });
        const dotIndex = Math.floor(this._time * 4) % 3;
        indicator.querySelectorAll('.dots circle').forEach((dot, index) => {
          dot.setAttribute('opacity', index === dotIndex ? '1' : '.42');
          dot.setAttribute('r', index === dotIndex ? '1.8' : '1.3');
        });
      }

      if (mode === 'portrait') this._animatePortrait(state);
      if (mode === 'grokbot') this._animateGrokbot(state);
    }

    _animatePortrait(state) {
      const head = this.shadowRoot.querySelector('.portrait-head');
      const backHair = this.shadowRoot.querySelector('.back-hair');
      const body = this.shadowRoot.querySelector('.body');
      const gaze = this.shadowRoot.querySelector('.eye-gaze');
      if (!head || !backHair || !body || !gaze) return;
      const breath = Math.sin(this._time * 1.7) * 2;
      const sway = Math.sin(this._time * 1.05) * 0.7;
      let headY = -breath * 0.24;
      if (state === 'listening') headY -= 0.55;
      else if (state === 'thinking') headY += 0.22;
      const headX = sway + (state === 'listening' ? -0.32 : 0);
      head.setAttribute('transform', `translate(${headX.toFixed(2)} ${headY.toFixed(2)})`);
      backHair.setAttribute('transform', `translate(${(headX * 0.35).toFixed(2)} ${headY.toFixed(2)})`);
      body.setAttribute('transform', `translate(${(sway * 0.45).toFixed(2)} ${(breath * 1.1).toFixed(2)})`);

      const leftBrow = this.shadowRoot.querySelector('.brow-left');
      const rightBrow = this.shadowRoot.querySelector('.brow-right');
      const browPaths = {
        idle: ['M15 42 C21 39 28 40 34 44', 'M52 49 C59 50 66 53 71 57'],
        happy: ['M15 41 C21 38 28 39 34 43', 'M52 48 C59 49 66 52 71 56'],
        listening: ['M15 40 C21 37 28 38 34 42', 'M52 47 C59 48 66 51 71 55'],
        thinking: ['M15 42 C21 39 28 39 34 42', 'M52 48 C59 48 66 51 71 55'],
        sad: ['M15 44 C21 41 27 37 34 38', 'M52 47 C59 49 66 54 71 58'],
        talking: ['M15 42 C21 39 28 40 34 44', 'M52 49 C59 50 66 53 71 57'],
      };
      const paths = browPaths[state] || browPaths.idle;
      leftBrow.setAttribute('d', paths[0]);
      rightBrow.setAttribute('d', paths[1]);

      this.shadowRoot.querySelectorAll('.blush').forEach((cheek, index) => {
        cheek.setAttribute('fill', state === 'happy' ? '#F5B9B9' : state === 'sad' ? '#F3D1CE' : '#F7C8C6');
        cheek.setAttribute('opacity', state === 'happy' ? '1' : '.94');
        if (state === 'happy') cheek.setAttribute('rx', index === 0 ? '6.5' : '7');
        else cheek.setAttribute('rx', index === 0 ? '6' : '6.5');
      });

      const closed = this._blinkLeft > 0 || state === 'happy';
      const openEyes = this.shadowRoot.querySelector('.open-eyes');
      const closedEyes = this.shadowRoot.querySelector('.closed-eyes');
      openEyes.style.display = closed ? 'none' : '';
      closedEyes.style.display = closed ? '' : 'none';
      let gazeX = 0, gazeY = 0;
      if (state === 'thinking') { gazeX = 1.3; gazeY = -1.35; }
      else if (state === 'sad') gazeY = 1;
      else if (state === 'listening') gazeX = Math.sin(this._time * 2) * 0.28;
      else if (state === 'talking') gazeX = Math.sin(this._time * 3.2) * 0.35;
      else { gazeX = Math.sin(this._time * 0.75) * 0.3; gazeY = Math.sin(this._time * 0.55) * 0.2; }
      gaze.setAttribute('transform', `translate(${gazeX.toFixed(2)} ${gazeY.toFixed(2)})`);
      if (!closed) {
        const eyes = this.shadowRoot.querySelectorAll('.portrait-eye');
        const height = state === 'listening' ? 10.8 : state === 'sad' ? 7.4 : 9.5;
        eyes.forEach(eye => eye.setAttribute('ry', String(height)));
      }
    }

    _animateGrokbot(state) {
      const eyes = this.shadowRoot.querySelectorAll('.classic-eye');
      if (!eyes.length) return;
      let dx = 0, dy = 0, width = 4.3, height = 8.6;
      if (state === 'listening') { width *= 1.18; height *= 1.24; dy = -1.8; }
      else if (state === 'thinking') { dx = Math.sin(this._time * 0.9) * 1.8 - 1.2; dy = -1.6; }
      else if (state === 'sad') { width *= 0.86; height *= 0.9; dy = 1.8; }
      else if (state === 'happy') { width *= 1.32; height *= 0.78; dy = -0.8; }
      else if (state === 'talking') height *= 0.85 + 0.4 * Math.abs(Math.sin(this._time * 9));
      else dx = Math.sin(this._time * 0.5) * 0.6;
      if (this._blinkLeft > 0) height = width * 0.12;
      eyes.forEach((eye, index) => {
        const side = index === 0 ? -1 : 1;
        eye.setAttribute('cx', String(50 + side * 15.5 + dx));
        eye.setAttribute('cy', String(52 + dy));
        eye.setAttribute('rx', String(width));
        eye.setAttribute('ry', String(height));
      });
    }
  }

  customElements.define('nova-avatar', NovaAvatar);
})();
