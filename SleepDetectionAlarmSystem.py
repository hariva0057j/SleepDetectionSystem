import React, { useState, useEffect, useRef } from 'react';

// --- DATA ---
const PROFILE = {
  name: "K. Hari Vamsi",
  title: "Computer Science & Engineering Student",
  email: "harivamsikandregula@gmail.com",
  phone: "+91 9989528669",
  linkedin: "linkedin.com/in/k-hari-vamsi",
  github: "github.com/harivamsikandregula-lab",
  about: "Passionate Computer Science & Engineering undergrad at Vishnu Institute of Technology. I specialize in software development, machine learning, and algorithm design.",
};

// Structured tokens for multi-color typing animation
const CONTENT_DATA = {
  about: [
    { text: "{\n", color: "text-white" },
    { text: '  "identity"', color: "text-cyan-400" }, { text: ": ", color: "text-white" }, { text: `"${PROFILE.name}"`, color: "text-amber-400" }, { text: ",\n", color: "text-white" },
    { text: '  "role"', color: "text-cyan-400" }, { text: ": ", color: "text-white" }, { text: `"${PROFILE.title}"`, color: "text-amber-400" }, { text: ",\n", color: "text-white" },
    { text: '  "status"', color: "text-cyan-400" }, { text: ": ", color: "text-white" }, { text: '"Active/Ready"', color: "text-emerald-400 font-bold" }, { text: ",\n", color: "text-white" },
    { text: '  "bio"', color: "text-cyan-400" }, { text: ": ", color: "text-white" }, { text: `"${PROFILE.about}"`, color: "text-amber-400" }, { text: ",\n", color: "text-white" },
    { text: '  "contact"', color: "text-cyan-400" }, { text: ": {\n", color: "text-white" },
    { text: '    "email"', color: "text-indigo-400" }, { text: ": ", color: "text-white" }, { text: `"${PROFILE.email}"`, color: "text-amber-400" }, { text: ",\n", color: "text-white" },
    { text: '    "github"', color: "text-indigo-400" }, { text: ": ", color: "text-white" }, { text: `"${PROFILE.github}"`, color: "text-amber-400" }, { text: ",\n", color: "text-white" },
    { text: '    "linkedin"', color: "text-indigo-400" }, { text: ": ", color: "text-white" }, { text: `"${PROFILE.linkedin}"`, color: "text-amber-400" }, { text: "\n", color: "text-white" },
    { text: "  }\n", color: "text-white" },
    { text: "}", color: "text-white" },
  ],
  skills: [
    { text: "root@hvamsi-os:~/skills$ tree\n", color: "text-slate-500" },
    { text: ".\n", color: "text-white" },
    { text: "├── ", color: "text-slate-600" }, { text: "Languages\n", color: "text-cyan-400 font-bold" },
    { text: "│   ├── ", color: "text-slate-600" }, { text: "Java (Core)\n", color: "text-white" },
    { text: "│   ├── ", color: "text-slate-600" }, { text: "Python\n", color: "text-white" },
    { text: "│   ├── ", color: "text-slate-600" }, { text: "C\n", color: "text-white" },
    { text: "│   └── ", color: "text-slate-600" }, { text: "SQL (MySQL)\n", color: "text-white" },
    { text: "├── ", color: "text-slate-600" }, { text: "Core_CS\n", color: "text-purple-400 font-bold" },
    { text: "│   ├── ", color: "text-slate-600" }, { text: "Data Structures & Algorithms\n", color: "text-white" },
    { text: "│   ├── ", color: "text-slate-600" }, { text: "OOP\n", color: "text-white" },
    { text: "│   ├── ", color: "text-slate-600" }, { text: "DBMS\n", color: "text-white" },
    { text: "│   └── ", color: "text-slate-600" }, { text: "Linux\n", color: "text-white" },
    { text: "├── ", color: "text-slate-600" }, { text: "Tools\n", color: "text-amber-400 font-bold" },
    { text: "│   ├── ", color: "text-slate-600" }, { text: "JDBC\n", color: "text-white" },
    { text: "│   ├── ", color: "text-slate-600" }, { text: "Git/GitHub\n", color: "text-white" },
    { text: "│   └── ", color: "text-slate-600" }, { text: "VS Code\n", color: "text-white" },
    { text: "└── ", color: "text-slate-600" }, { text: "Automation\n", color: "text-emerald-400 font-bold" },
    { text: "    ├── ", color: "text-slate-600" }, { text: "Zapier\n", color: "text-white" },
    { text: "    └── ", color: "text-slate-600" }, { text: "n8n", color: "text-white" },
  ],
  projects: [
    { text: "[OCT 2025] ", color: "text-purple-500" }, { text: "DROWSINESS ALERT SYSTEM\n", color: "text-white font-bold" },
    { text: "> Designed EAR system for real-time sleep detection.\n", color: "text-slate-400" },
    { text: "> Tech: ", color: "text-emerald-500" }, { text: "Python, OpenCV, MediaPipe, ML\n\n", color: "text-cyan-300" },

    { text: "[SEP 2025] ", color: "text-purple-500" }, { text: "CLASS TRACK PRO\n", color: "text-white font-bold" },
    { text: "> Responsive web app for attendance tracking.\n", color: "text-slate-400" },
    { text: "> Tech: ", color: "text-emerald-500" }, { text: "HTML, CSS, JavaScript, Netlify\n\n", color: "text-cyan-300" },

    { text: "[MAR 2025] ", color: "text-purple-500" }, { text: "HOTEL MANAGEMENT SYSTEM\n", color: "text-white font-bold" },
    { text: "> Scalable backend with secure JDBC connectivity.\n", color: "text-slate-400" },
    { text: "> Tech: ", color: "text-emerald-500" }, { text: "Java, JDBC, MySQL", color: "text-cyan-300" },
  ],
  experience: [
    { text: "LOG_EVENT: 2025-01 ", color: "text-slate-500" }, { text: "SIH 2025 - PARTICIPANT\n", color: "text-white font-bold" },
    { text: "INFO: Built quantum-inspired secure email prototype (QMail).\n\n", color: "text-emerald-400" },
    
    { text: "LOG_EVENT: 2024-06 ", color: "text-slate-500" }, { text: "FAILATHON 2024 - MENTOR\n", color: "text-white font-bold" },
    { text: "INFO: Coached 5+ teams on Git workflows and DSA.\n\n", color: "text-amber-400" },
    
    { text: "LOG_EVENT: 2023-11 ", color: "text-slate-500" }, { text: "FAILATHON 2023 - PARTICIPANT\n", color: "text-white font-bold" },
    { text: "INFO: Built logic-based prototype under 24-hr deadline.", color: "text-emerald-400" },
  ],
  education: [
    { text: "# ACADEMIC_RECORD\n", color: "text-white font-bold text-lg" },
    { text: "1. VISHNU INSTITUTE OF TECHNOLOGY\n", color: "text-cyan-400 font-bold" },
    { text: "   B.Tech CSE [2023-2027] | ", color: "text-slate-400" }, { text: "GPA: 9.0/10\n", color: "text-emerald-400 font-bold" },
    { text: "2. SRI CHAITANYA COLLEGE\n", color: "text-cyan-400 font-bold" },
    { text: "   Intermediate [2021-2023] | ", color: "text-slate-400" }, { text: "Score: 96.3%\n\n", color: "text-emerald-400 font-bold" },
    
    { text: "# CERTIFICATIONS\n", color: "text-white font-bold text-lg" },
    { text: "- ", color: "text-white" }, { text: "Privacy & Security in Online Social Media\n", color: "text-amber-400" },
    { text: "- ", color: "text-white" }, { text: "Programming in Java (IIT Kharagpur)\n", color: "text-amber-400" },
    { text: "- ", color: "text-white" }, { text: "StemExpo Participation (Aug 2025)", color: "text-amber-400" },
  ]
};

// --- COMPONENTS ---

const MultiColorTypewriter = ({ tokens, delay = 5, onComplete }) => {
  const [displayedTokens, setDisplayedTokens] = useState([]);
  const [currentTokenIdx, setCurrentTokenIdx] = useState(0);
  const [currentCharIdx, setCurrentCharIdx] = useState(0);

  useEffect(() => {
    // Reset state when tokens change
    setDisplayedTokens([]);
    setCurrentTokenIdx(0);
    setCurrentCharIdx(0);
  }, [tokens]);

  useEffect(() => {
    if (currentTokenIdx >= tokens.length) {
      if (onComplete) onComplete();
      return;
    }

    const currentToken = tokens[currentTokenIdx];
    const typingInterval = setInterval(() => {
      if (currentCharIdx < currentToken.text.length) {
        // We update the list of displayed tokens
        setDisplayedTokens(prev => {
          const newTokens = [...prev];
          if (!newTokens[currentTokenIdx]) {
            newTokens[currentTokenIdx] = { ...currentToken, text: "" };
          }
          newTokens[currentTokenIdx].text = currentToken.text.slice(0, currentCharIdx + 1);
          return newTokens;
        });
        setCurrentCharIdx(prev => prev + 1);
      } else {
        clearInterval(typingInterval);
        setCurrentTokenIdx(prev => prev + 1);
        setCurrentCharIdx(0);
      }
    }, delay);

    return () => clearInterval(typingInterval);
  }, [currentTokenIdx, currentCharIdx, tokens, delay, onComplete]);

  return (
    <pre className="whitespace-pre-wrap font-mono break-words leading-relaxed tracking-tight text-sm md:text-base">
      {displayedTokens.map((token, i) => (
        <span key={i} className={token.color}>{token.text}</span>
      ))}
      <span className="inline-block w-2.5 h-5 bg-emerald-500 ml-1 animate-pulse align-middle"></span>
    </pre>
  );
};

const TerminalPrompt = ({ path = "~" }) => (
  <span className="font-bold">
    <span className="text-emerald-500">guest@hvamsi-os</span>
    <span className="text-white">:</span>
    <span className="text-blue-400">{path}</span>
    <span className="text-white">$ </span>
  </span>
);

export default function App() {
  const [booting, setBooting] = useState(true);
  const [activeTab, setActiveTab] = useState('about');
  const [commandTyped, setCommandTyped] = useState("");
  const [isCommandDone, setIsCommandDone] = useState(false);
  const [renderKey, setRenderKey] = useState(0);

  const tabs = [
    { id: 'about', cmd: './whoami.sh', path: '~' },
    { id: 'skills', cmd: 'ls -R ./skills', path: '~/skills' },
    { id: 'projects', cmd: './run_projects.bin', path: '~/projects' },
    { id: 'experience', cmd: 'cat /var/log/events.log', path: '~' },
    { id: 'education', cmd: 'cat education.md', path: '~' }
  ];

  useEffect(() => {
    const timer = setTimeout(() => setBooting(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  // Handle Initial & Tab Change Command Typing
  useEffect(() => {
    if (booting) return;
    
    setIsCommandDone(false);
    setCommandTyped("");
    const targetCmd = tabs.find(t => t.id === activeTab).cmd;
    
    let i = 0;
    const interval = setInterval(() => {
      setCommandTyped(targetCmd.slice(0, i + 1));
      i++;
      if (i >= targetCmd.length) {
        clearInterval(interval);
        setTimeout(() => {
            setIsCommandDone(true);
            setRenderKey(prev => prev + 1); 
        }, 150);
      }
    }, 35);

    return () => clearInterval(interval);
  }, [activeTab, booting]);

  if (booting) {
    return (
      <div className="min-h-screen bg-black text-emerald-500 font-mono p-10 flex flex-col justify-end">
        <style dangerouslySetInnerHTML={{__html: `@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;700&display=swap'); body { font-family: 'Fira Code', monospace; }`}} />
        <div className="space-y-1 animate-pulse text-sm md:text-base">
          <p className="text-emerald-700">[    0.000000] Initializing HVAMSI-OS kernel...</p>
          <p className="text-emerald-600">[    0.452123] Security Check: PASSED</p>
          <p className="text-emerald-500">[    0.892110] Loading System Modules... OK</p>
          <p className="text-emerald-400">[    1.200000] Opening Terminal Interface v2.5.0</p>
        </div>
      </div>
    );
  }

  const currentTabInfo = tabs.find(t => t.id === activeTab);

  return (
    <div className="min-h-screen bg-[#050505] text-slate-300 font-mono flex flex-col relative overflow-hidden">
      
      {/* Visual FX: CRT & Scanlines */}
      <style dangerouslySetInnerHTML={{__html: `
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;700&display=swap');
        
        .scanline {
          width: 100%; height: 100px; z-index: 100;
          background: linear-gradient(0deg, rgba(0,0,0,0) 0%, rgba(16, 185, 129, 0.03) 50%, rgba(0,0,0,0) 100%);
          position: absolute; bottom: 100%; animation: scanline 10s linear infinite; pointer-events: none;
        }
        @keyframes scanline { 0% { bottom: 100%; } 100% { bottom: -100px; } }

        .crt-overlay {
          position: absolute; inset: 0; pointer-events: none; z-index: 90;
          background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
          background-size: 100% 3px, 3px 100%;
        }

        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #0a0a0a; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #166534; }
      `}} />

      <div className="scanline"></div>
      <div className="crt-overlay"></div>

      {/* HEADER BAR */}
      <header className="bg-[#0a0a0a] border-b border-emerald-900/40 p-3 flex items-center justify-between shrink-0 z-50">
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-red-600/50"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-600/50"></div>
          <div className="w-3 h-3 rounded-full bg-emerald-600/50"></div>
        </div>
        <div className="text-[11px] font-bold text-emerald-800 tracking-widest uppercase">
          HVAMSI@REMOTE: ~ (SSH)
        </div>
        <div className="hidden md:flex gap-4 text-[10px] text-emerald-900 font-bold">
          <span>PORT: 8080</span>
          <span>STATUS: ONLINE</span>
        </div>
      </header>

      {/* COMMAND MENU (TABS) */}
      <nav className="bg-[#080808] border-b border-emerald-900/20 p-2 overflow-x-auto z-50 flex gap-2 no-scrollbar scroll-smooth">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => activeTab !== tab.id && setActiveTab(tab.id)}
            className={`px-4 py-1.5 text-xs font-bold border transition-all duration-200 rounded-sm whitespace-nowrap ${
              activeTab === tab.id 
                ? 'bg-emerald-950/30 text-emerald-400 border-emerald-500/40 shadow-[0_0_10px_rgba(16,185,129,0.1)]' 
                : 'text-emerald-900 border-transparent hover:text-emerald-600 hover:bg-emerald-950/10'
            }`}
          >
            {tab.id.toUpperCase()}.SH
          </button>
        ))}
      </nav>

      {/* MAIN CONSOLE AREA */}
      <main className="flex-1 p-6 md:p-12 overflow-y-auto custom-scrollbar z-40 bg-[radial-gradient(circle_at_center,_rgba(16,185,129,0.01)_0%,_transparent_80%)]">
        <div className="max-w-4xl mx-auto w-full">
            
            {/* Command Line Input */}
            <div className="flex items-center gap-2 text-sm md:text-lg mb-8">
                <TerminalPrompt path={currentTabInfo.path} />
                <span className="text-white font-bold tracking-tight">{commandTyped}</span>
                {!isCommandDone && <span className="w-2.5 h-6 bg-emerald-500 animate-pulse"></span>}
            </div>

            {/* Structured Content Output */}
            {isCommandDone && (
              <div key={renderKey} className="selection:bg-emerald-500/30">
                <MultiColorTypewriter 
                  tokens={CONTENT_DATA[activeTab]} 
                  delay={10} 
                />
              </div>
            )}
        </div>
      </main>

      {/* SYSTEM STATUS FOOTER */}
      <footer className="bg-[#0a0a0a] border-t border-emerald-900/40 p-2 px-6 text-[11px] flex justify-between text-emerald-900 font-bold z-50 shrink-0">
        <div className="flex gap-6">
          <span className="text-emerald-800">UTF-8</span>
          <span className="hidden sm:inline">TTY: S001</span>
        </div>
        <div className="flex gap-5 items-center">
            <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> 
                CPU_LOAD: 0.04
            </span>
            <span className="hidden sm:inline">MEM: 512MB / 4096MB</span>
            <span className="text-white opacity-40 uppercase tracking-tighter">Terminal V2.5</span>
        </div>
      </footer>
    </div>
  );
}

