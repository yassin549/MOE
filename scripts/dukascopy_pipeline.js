const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { spawnSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(ROOT, "data");
const RAW_DIR = path.join(DATA_DIR, "raw");
const PROCESSED_DIR = path.join(DATA_DIR, "processed");
const REPORTS_DIR = path.join(ROOT, "reports");

const CONFIG = {
  timeframe: "m1",
  from: "2021-04-27",
  toExclusive: "2026-04-28",
  instruments: [
    { symbol: "US100", instrumentId: "usatechidxusd" },
    { symbol: "US500", instrumentId: "usa500idxusd" },
  ],
};

const MONTH_NAMES = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const DAY_NAMES = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function runCommand(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: ROOT,
    stdio: "inherit",
    shell: false,
    ...options,
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with exit code ${result.status}`);
  }
}

function spawnCommand(command, args, options = {}) {
  if (process.platform === "win32") {
    return spawnSync("cmd.exe", ["/c", command, ...args], options);
  }
  return spawnSync(command, args, options);
}

function isoWeekNumber(date) {
  const utcDate = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  const dayNum = utcDate.getUTCDay() || 7;
  utcDate.setUTCDate(utcDate.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(utcDate.getUTCFullYear(), 0, 1));
  return Math.ceil((((utcDate - yearStart) / 86400000) + 1) / 7);
}

function formatIsoMinute(date) {
  return date.toISOString().slice(0, 16) + ":00Z";
}

function addMonthsUtc(date, months) {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + months, date.getUTCDate()));
}

function formatDateOnly(date) {
  return date.toISOString().slice(0, 10);
}

function buildMonthlyChunks(fromInclusive, toExclusive) {
  const chunks = [];
  let cursor = new Date(`${fromInclusive}T00:00:00.000Z`);
  const end = new Date(`${toExclusive}T00:00:00.000Z`);

  while (cursor < end) {
    const nextMonthStart = new Date(Date.UTC(cursor.getUTCFullYear(), cursor.getUTCMonth() + 1, 1));
    const chunkEnd = nextMonthStart < end ? nextMonthStart : end;
    chunks.push({
      from: formatDateOnly(cursor),
      to: formatDateOnly(chunkEnd),
    });
    cursor = chunkEnd;
  }

  return chunks;
}

function daysBetween(fromDate, toDate) {
  const from = new Date(`${fromDate}T00:00:00.000Z`);
  const to = new Date(`${toDate}T00:00:00.000Z`);
  return Math.round((to - from) / 86400000);
}

function splitDateRange(fromDate, toDate) {
  const from = new Date(`${fromDate}T00:00:00.000Z`);
  const to = new Date(`${toDate}T00:00:00.000Z`);
  const spanDays = Math.round((to - from) / 86400000);
  const leftDays = Math.max(1, Math.floor(spanDays / 2));
  const middle = new Date(from.getTime() + leftDays * 86400000);
  const middleDate = formatDateOnly(middle);
  return [
    { from: fromDate, to: middleDate },
    { from: middleDate, to: toDate },
  ];
}

function downloadDateRange({ symbol, instrumentId, fromDate, toDate, chunkDir }) {
  const targetFile = path.join(
    chunkDir,
    `${symbol}_${CONFIG.timeframe}_${fromDate}_${toDate}.csv`
  );

  if (fs.existsSync(targetFile) && fs.statSync(targetFile).size > 0) {
    console.log(`Reusing chunk ${path.basename(targetFile)}`);
    return [targetFile];
  }

  const args = [
    "dukascopy-node@latest",
    "-i",
    instrumentId,
    "-from",
    fromDate,
    "-to",
    toDate,
    "-t",
    CONFIG.timeframe,
    "-f",
    "csv",
    "-v",
    "-dir",
    ".",
    "-r",
    "3",
    "-rp",
    "2000",
    "-re",
  ];

  console.log(`Downloading ${symbol} chunk ${fromDate} -> ${toDate}...`);
  const beforeFiles = new Set(fs.readdirSync(chunkDir));
  const result = spawnCommand("npx", args, {
    cwd: chunkDir,
    encoding: "utf8",
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status === 0) {
    const afterFiles = fs
      .readdirSync(chunkDir)
      .filter((fileName) => fileName.toLowerCase().endsWith(".csv"));
    const newFiles = afterFiles.filter((fileName) => !beforeFiles.has(fileName));
    const candidateFiles = (newFiles.length ? newFiles : afterFiles)
      .map((fileName) => ({
        fileName,
        mtimeMs: fs.statSync(path.join(chunkDir, fileName)).mtimeMs,
      }))
      .sort((a, b) => b.mtimeMs - a.mtimeMs);

    if (candidateFiles.length === 0) {
      throw new Error(`Download completed but no CSV file was found for ${instrumentId} chunk ${fromDate} -> ${toDate}`);
    }

    const downloadedFile = path.join(chunkDir, candidateFiles[0].fileName);
    if (downloadedFile !== targetFile) {
      fs.renameSync(downloadedFile, targetFile);
    }
    return [targetFile];
  }

  if (daysBetween(fromDate, toDate) <= 1) {
    throw new Error(`npx ${args.join(" ")} failed.\n${result.stderr || result.stdout || ""}`);
  }

  console.warn(`Chunk failed for ${symbol} ${fromDate} -> ${toDate}; splitting range and retrying.`);
  const [left, right] = splitDateRange(fromDate, toDate);
  return [
    ...downloadDateRange({ symbol, instrumentId, fromDate: left.from, toDate: left.to, chunkDir }),
    ...downloadDateRange({ symbol, instrumentId, fromDate: right.from, toDate: right.to, chunkDir }),
  ];
}

function sessionBucket(hour) {
  if (hour >= 0 && hour < 7) return "asia";
  if (hour >= 7 && hour < 13) return "europe";
  if (hour >= 13 && hour < 17) return "us_overlap";
  if (hour >= 17 && hour < 21) return "us";
  return "post_us";
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function average(values) {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function topEntries(mapObject, limit = 10, valueKey = "value") {
  return Object.entries(mapObject)
    .map(([key, value]) => ({ key, [valueKey]: value }))
    .sort((a, b) => b[valueKey] - a[valueKey])
    .slice(0, limit);
}

function increment(mapObject, key, amount = 1) {
  mapObject[key] = (mapObject[key] || 0) + amount;
}

function addStatBucket(target, key, value) {
  if (!target[key]) {
    target[key] = { count: 0, absReturnSum: 0, volumeSum: 0 };
  }
  target[key].count += 1;
  target[key].absReturnSum += Math.abs(value.returnPct || 0);
  target[key].volumeSum += value.volume || 0;
}

async function cleanAndAnalyzeInstrument({ symbol, instrumentId }) {
  const rawFile = path.join(RAW_DIR, `${symbol}_${CONFIG.timeframe}_raw.csv`);
  const cleanFile = path.join(PROCESSED_DIR, `${symbol}_${CONFIG.timeframe}_clean.csv`);
  const summaryFile = path.join(REPORTS_DIR, `${symbol.toLowerCase()}_${CONFIG.timeframe}_summary.json`);

  const input = fs.createReadStream(rawFile);
  const rl = readline.createInterface({ input, crlfDelay: Infinity });
  const output = fs.createWriteStream(cleanFile);

  const header = [
    "symbol",
    "instrument_id",
    "timestamp_utc",
    "timestamp_ms",
    "year",
    "quarter",
    "month",
    "month_name",
    "iso_week",
    "day_of_month",
    "day_of_week",
    "day_name",
    "hour_utc",
    "minute_utc",
    "session_utc",
    "is_month_start",
    "is_month_end",
    "minutes_since_prev_bar",
    "is_gap_after_prev_bar",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1m_pct",
    "log_return_1m",
    "range_pct",
    "body_pct"
  ];
  output.write(header.join(",") + "\n");

  let isHeader = true;
  let prevTimestamp = null;
  let prevClose = null;
  let prevDateKey = null;
  let rowsRead = 0;
  let rowsWritten = 0;
  let duplicatesDropped = 0;
  let invalidRowsDropped = 0;
  let nonAscendingDropped = 0;
  let zeroVolumeDropped = 0;
  let gapCount = 0;
  const gapMinutes = [];
  const rowsPerDay = {};
  const byHour = {};
  const byWeekday = {};
  const byMonth = {};
  const tradingDays = new Set();
  const timestampsSeen = new Set();

  for await (const line of rl) {
    if (isHeader) {
      isHeader = false;
      continue;
    }
    if (!line.trim()) continue;
    rowsRead += 1;

    const parts = line.split(",");
    if (parts.length < 6) {
      invalidRowsDropped += 1;
      continue;
    }

    const [timestampRaw, openRaw, highRaw, lowRaw, closeRaw, volumeRaw] = parts;
    const timestampMs = Number(timestampRaw);
    const open = Number(openRaw);
    const high = Number(highRaw);
    const low = Number(lowRaw);
    const close = Number(closeRaw);
    const volume = Number(volumeRaw);

    const hasFiniteNumbers = [timestampMs, open, high, low, close, volume].every(Number.isFinite);
    const validOhlc =
      low <= Math.min(open, high, close) &&
      high >= Math.max(open, low, close) &&
      open > 0 &&
      high > 0 &&
      low > 0 &&
      close > 0;

    if (!hasFiniteNumbers || !validOhlc) {
      invalidRowsDropped += 1;
      continue;
    }

    if (volume <= 0) {
      zeroVolumeDropped += 1;
      continue;
    }

    if (timestampsSeen.has(timestampMs)) {
      duplicatesDropped += 1;
      continue;
    }
    timestampsSeen.add(timestampMs);

    if (prevTimestamp !== null && timestampMs <= prevTimestamp) {
      nonAscendingDropped += 1;
      continue;
    }

    const date = new Date(timestampMs);
    const year = date.getUTCFullYear();
    const month = date.getUTCMonth() + 1;
    const dayOfMonth = date.getUTCDate();
    const dayOfWeek = date.getUTCDay();
    const hour = date.getUTCHours();
    const minute = date.getUTCMinutes();
    const isoWeek = isoWeekNumber(date);
    const quarter = Math.floor((month - 1) / 3) + 1;
    const dayKey = date.toISOString().slice(0, 10);
    const nextDay = new Date(Date.UTC(year, date.getUTCMonth(), dayOfMonth + 1));
    const isMonthEnd = nextDay.getUTCDate() === 1 ? 1 : 0;
    const isMonthStart = dayOfMonth === 1 ? 1 : 0;

    let minutesSincePrevBar = "";
    let isGapAfterPrevBar = 0;
    if (prevTimestamp !== null) {
      const diffMinutes = (timestampMs - prevTimestamp) / 60000;
      minutesSincePrevBar = diffMinutes;
      if (diffMinutes > 1) {
        isGapAfterPrevBar = 1;
        gapCount += 1;
        gapMinutes.push(diffMinutes);
      }
    }

    const returnPct = prevClose === null ? "" : ((close / prevClose) - 1) * 100;
    const logReturn = prevClose === null ? "" : Math.log(close / prevClose);
    const rangePct = ((high - low) / close) * 100;
    const bodyPct = ((close - open) / open) * 100;

    const row = [
      symbol,
      instrumentId,
      formatIsoMinute(date),
      timestampMs,
      year,
      quarter,
      month,
      MONTH_NAMES[month - 1],
      isoWeek,
      dayOfMonth,
      dayOfWeek,
      DAY_NAMES[dayOfWeek],
      hour,
      minute,
      sessionBucket(hour),
      isMonthStart,
      isMonthEnd,
      minutesSincePrevBar,
      isGapAfterPrevBar,
      open,
      high,
      low,
      close,
      volume,
      returnPct,
      logReturn,
      rangePct,
      bodyPct,
    ];

    output.write(row.join(",") + "\n");
    rowsWritten += 1;
    tradingDays.add(dayKey);
    increment(rowsPerDay, dayKey);

    addStatBucket(byHour, String(hour).padStart(2, "0"), { returnPct, volume });
    addStatBucket(byWeekday, DAY_NAMES[dayOfWeek], { returnPct, volume });
    addStatBucket(byMonth, MONTH_NAMES[month - 1], { returnPct, volume });

    prevTimestamp = timestampMs;
    prevClose = close;
    prevDateKey = dayKey;
  }

  await new Promise((resolve) => output.end(resolve));

  const dayCounts = Object.values(rowsPerDay);
  const summary = {
    symbol,
    instrumentId,
    timeframe: CONFIG.timeframe,
    source: "Dukascopy",
    dateRange: {
      fromInclusive: CONFIG.from,
      toExclusive: CONFIG.toExclusive,
    },
    files: {
      raw: rawFile,
      cleaned: cleanFile,
    },
    rowCounts: {
      rawRowsRead: rowsRead,
      cleanedRowsWritten: rowsWritten,
      duplicatesDropped,
      invalidRowsDropped,
      nonAscendingDropped,
      zeroVolumeDropped,
    },
    tradingDays: tradingDays.size,
    barsPerTradingDay: {
      average: average(dayCounts),
      median: median(dayCounts),
      min: dayCounts.length ? Math.min(...dayCounts) : null,
      max: dayCounts.length ? Math.max(...dayCounts) : null,
    },
    gaps: {
      count: gapCount,
      averageMinutes: average(gapMinutes),
      medianMinutes: median(gapMinutes),
      maxMinutes: gapMinutes.length ? Math.max(...gapMinutes) : null,
    },
    busiestDays: topEntries(rowsPerDay, 10, "bars"),
    byHourUtc: Object.fromEntries(
      Object.entries(byHour).map(([key, value]) => [
        key,
        {
          bars: value.count,
          avgAbsReturnPct: value.count ? value.absReturnSum / value.count : 0,
          avgVolume: value.count ? value.volumeSum / value.count : 0,
        },
      ])
    ),
    byWeekday: Object.fromEntries(
      Object.entries(byWeekday).map(([key, value]) => [
        key,
        {
          bars: value.count,
          avgAbsReturnPct: value.count ? value.absReturnSum / value.count : 0,
          avgVolume: value.count ? value.volumeSum / value.count : 0,
        },
      ])
    ),
    byMonth: Object.fromEntries(
      Object.entries(byMonth).map(([key, value]) => [
        key,
        {
          bars: value.count,
          avgAbsReturnPct: value.count ? value.absReturnSum / value.count : 0,
          avgVolume: value.count ? value.volumeSum / value.count : 0,
        },
      ])
    ),
  };

  fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));
  return summary;
}

function formatTopSection(title, section) {
  const lines = [`## ${title}`];
  for (const [key, value] of Object.entries(section)) {
    lines.push(
      `- ${key}: bars=${value.bars}, avg_abs_return_pct=${value.avgAbsReturnPct.toFixed(6)}, avg_volume=${value.avgVolume.toFixed(6)}`
    );
  }
  return lines.join("\n");
}

function writeCombinedReport(summaries) {
  const reportPath = path.join(REPORTS_DIR, "data_pipeline_report.md");
  const lines = [
    "# Dukascopy US100 / US500 Data Report",
    "",
    `Source: Dukascopy minute bars downloaded for ${CONFIG.from} through ${CONFIG.toExclusive} (exclusive end date).`,
    `Instruments: ${CONFIG.instruments.map((item) => `${item.symbol} (${item.instrumentId})`).join(", ")}`,
    "",
    "## Cleaning policy",
    "- Removed malformed OHLCV rows.",
    "- Removed zero-volume rows.",
    "- Removed duplicate timestamps.",
    "- Preserved genuine market-session gaps and exposed them with `minutes_since_prev_bar` and `is_gap_after_prev_bar`.",
    "- Enriched each bar with UTC calendar fields for direct model feature engineering.",
    "",
  ];

  for (const summary of summaries) {
    lines.push(`## ${summary.symbol}`);
    lines.push(`- Clean rows: ${summary.rowCounts.cleanedRowsWritten}`);
    lines.push(`- Raw rows read: ${summary.rowCounts.rawRowsRead}`);
    lines.push(`- Duplicates dropped: ${summary.rowCounts.duplicatesDropped}`);
    lines.push(`- Invalid rows dropped: ${summary.rowCounts.invalidRowsDropped}`);
    lines.push(`- Zero-volume rows dropped: ${summary.rowCounts.zeroVolumeDropped}`);
    lines.push(`- Trading days: ${summary.tradingDays}`);
    lines.push(
      `- Bars per trading day: avg=${summary.barsPerTradingDay.average?.toFixed(2)}, median=${summary.barsPerTradingDay.median?.toFixed(2)}, min=${summary.barsPerTradingDay.min}, max=${summary.barsPerTradingDay.max}`
    );
    lines.push(
      `- Gaps: count=${summary.gaps.count}, avg_minutes=${summary.gaps.averageMinutes?.toFixed(2) ?? "n/a"}, median_minutes=${summary.gaps.medianMinutes?.toFixed(2) ?? "n/a"}, max_minutes=${summary.gaps.maxMinutes ?? "n/a"}`
    );
    lines.push("");
    lines.push(formatTopSection(`${summary.symbol} Hourly Profile (UTC)`, summary.byHourUtc));
    lines.push("");
    lines.push(formatTopSection(`${summary.symbol} Weekday Profile`, summary.byWeekday));
    lines.push("");
    lines.push(formatTopSection(`${summary.symbol} Monthly Profile`, summary.byMonth));
    lines.push("");
  }

  fs.writeFileSync(reportPath, lines.join("\n"));
}

function downloadInstrument({ symbol, instrumentId }) {
  const rawFile = path.join(RAW_DIR, `${symbol}_${CONFIG.timeframe}_raw.csv`);
  if (fs.existsSync(rawFile) && fs.statSync(rawFile).size > 0) {
    console.log(`Skipping download for ${symbol}; raw file already exists at ${rawFile}`);
    return;
  }

  const chunkDir = path.join(RAW_DIR, "chunks", symbol);
  ensureDir(chunkDir);
  const chunks = buildMonthlyChunks(CONFIG.from, CONFIG.toExclusive);
  const chunkFiles = [];

  for (const chunk of chunks) {
    chunkFiles.push(
      ...downloadDateRange({
        symbol,
        instrumentId,
        fromDate: chunk.from,
        toDate: chunk.to,
        chunkDir,
      })
    );
  }

  fs.writeFileSync(rawFile, "");
  let wroteHeader = false;
  for (const chunkFile of chunkFiles) {
    const content = fs.readFileSync(chunkFile, "utf8");
    const lines = content.split(/\r?\n/).filter(Boolean);
    if (lines.length === 0) continue;
    if (!wroteHeader) {
      fs.appendFileSync(rawFile, lines.join("\n") + "\n");
      wroteHeader = true;
      continue;
    }
    fs.appendFileSync(rawFile, lines.slice(1).join("\n") + "\n");
  }
  console.log(`Saved merged raw CSV to ${rawFile}`);
}

async function main() {
  ensureDir(RAW_DIR);
  ensureDir(PROCESSED_DIR);
  ensureDir(REPORTS_DIR);

  for (const instrument of CONFIG.instruments) {
    downloadInstrument(instrument);
  }

  const summaries = [];
  for (const instrument of CONFIG.instruments) {
    console.log(`Cleaning and analyzing ${instrument.symbol}...`);
    const summary = await cleanAndAnalyzeInstrument(instrument);
    summaries.push(summary);
  }

  writeCombinedReport(summaries);
  console.log("Pipeline complete.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
