import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(import.meta.dirname, "..");
const outputDir = path.join(root, "outputs", "weather_analysis");
const previewDir = path.join(outputDir, "previews");
await fs.mkdir(previewDir, { recursive: true });

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"' && inQuotes && next === '"') {
      field += '"';
      i += 1;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(field);
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function coerce(value) {
  if (value === "") return null;
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}

async function readCsvMatrix(fileName) {
  const text = await fs.readFile(path.join(outputDir, fileName), "utf8");
  return parseCsv(text).map((row, rowIndex) => (rowIndex === 0 ? row : row.map(coerce)));
}

function colName(num) {
  let name = "";
  while (num > 0) {
    const mod = (num - 1) % 26;
    name = String.fromCharCode(65 + mod) + name;
    num = Math.floor((num - mod) / 26);
  }
  return name;
}

function rangeFor(matrix) {
  return `A1:${colName(matrix[0].length)}${matrix.length}`;
}

function writeSheet(workbook, sheetName, matrix, tableName, numberFormats = {}) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  sheet.getRange(rangeFor(matrix)).values = matrix;
  sheet.getRange(`A1:${colName(matrix[0].length)}1`).format = {
    fill: "#205A6B",
    font: { bold: true, color: "#FFFFFF" },
  };
  sheet.getRange(rangeFor(matrix)).format.borders = {
    preset: "inside",
    style: "thin",
    color: "#D9E2E7",
  };
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add(rangeFor(matrix), true, tableName);
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  for (const [columnIndex, formatCode] of Object.entries(numberFormats)) {
    const col = colName(Number(columnIndex));
    sheet.getRange(`${col}2:${col}${matrix.length}`).format.numberFormat = formatCode;
  }
  sheet.getRange(rangeFor(matrix)).format.autofitColumns();
  return sheet;
}

const manifest = JSON.parse(await fs.readFile(path.join(outputDir, "manifest.json"), "utf8"));
const workbook = Workbook.create();

const summary = workbook.worksheets.add("Summary");
summary.showGridLines = false;
summary.getRange("A1:F1").merge();
summary.getRange("A1").values = [["Weather Data Analysis"]];
summary.getRange("A1").format = {
  fill: "#183B47",
  font: { bold: true, color: "#FFFFFF", size: 16 },
};

summary.getRange("A3:B11").values = [
  ["Paris period", `${manifest.analysis.paris.date_start} to ${manifest.analysis.paris.date_end}`],
  ["Mean temperature (C)", manifest.analysis.paris.mean_temp_c],
  ["Max temperature (C)", manifest.analysis.paris.max_temp_c],
  ["Min temperature (C)", manifest.analysis.paris.min_temp_c],
  ["Total precipitation (mm)", manifest.analysis.paris.total_precip_mm],
  ["Mean 10m wind (m/s)", manifest.analysis.paris.mean_wind10_mps],
  ["Hottest hour UTC", manifest.analysis.paris.hottest_hour_utc],
  ["Windiest hour UTC", manifest.analysis.paris.windiest_hour_utc],
  ["Source file", "Paris ZIP + GRIB/NetCDF streams"],
];
summary.getRange("D3:E11").values = [
  ["Germany period", `${manifest.analysis.germany.date_start} to ${manifest.analysis.germany.date_end}`],
  ["Mean 10m wind (m/s)", manifest.analysis.germany.mean_wind10_mps],
  ["Max grid wind (m/s)", manifest.analysis.germany.max_area_wind10_mps],
  ["Windiest day", manifest.analysis.germany.windiest_day],
  ["Windiest month", manifest.analysis.germany.windiest_month],
  ["Calmest month", manifest.analysis.germany.calmest_month],
  ["Top grid latitude", manifest.analysis.germany.highest_mean_grid_cell.latitude],
  ["Top grid longitude", manifest.analysis.germany.highest_mean_grid_cell.longitude],
  ["Top grid mean wind (m/s)", manifest.analysis.germany.highest_mean_grid_cell.mean_wind10_mps],
];
summary.getRange("A3:B11").format.borders = { preset: "all", style: "thin", color: "#C9D6DC" };
summary.getRange("D3:E11").format.borders = { preset: "all", style: "thin", color: "#C9D6DC" };
summary.getRange("A3:A11").format = { fill: "#E8F2F5", font: { bold: true } };
summary.getRange("D3:D11").format = { fill: "#E8F2F5", font: { bold: true } };
summary.getRange("B4:B7").format.numberFormat = "0.00";
summary.getRange("E4:E5").format.numberFormat = "0.00";
summary.getRange("E9:E11").format.numberFormat = "0.00";
summary.getRange("A:E").format.autofitColumns();

const parisDaily = await readCsvMatrix("paris_daily_summary.csv");
const parisHourly = await readCsvMatrix("paris_hourly_area_mean.csv");
const parisGrid = await readCsvMatrix("paris_hourly_grid.csv");
const germanyDaily = await readCsvMatrix("germany_daily_summary.csv");
const germanyMonthly = await readCsvMatrix("germany_monthly_summary.csv");
const germanyHourly = await readCsvMatrix("germany_hourly_area_mean.csv");
const germanyGrid = await readCsvMatrix("germany_grid_cell_summary.csv");

writeSheet(workbook, "Paris Daily", parisDaily, "ParisDaily", {
  2: "0.00",
  3: "0.00",
  4: "0.00",
  6: "0.00",
  7: "0.00",
  8: "0.00",
});
writeSheet(workbook, "Paris Hourly Mean", parisHourly, "ParisHourlyMean", {
  2: "0.00",
  8: "0.00",
  9: "0.00",
  10: "0.00",
});
writeSheet(workbook, "Paris Hourly Grid", parisGrid, "ParisHourlyGrid", {
  4: "0.00",
  10: "0.00",
  13: "0.00",
  17: "0.00",
});
writeSheet(workbook, "Germany Daily", germanyDaily, "GermanyDaily", {
  2: "0.00",
  3: "0.00",
  4: "0.00",
  5: "0.00",
});
writeSheet(workbook, "Germany Monthly", germanyMonthly, "GermanyMonthly", {
  2: "0.00",
  3: "0.00",
  4: "0.00",
  5: "0.00",
});
writeSheet(workbook, "Germany Hourly Mean", germanyHourly, "GermanyHourlyMean", {
  2: "0.00",
  3: "0.00",
  4: "0.00",
  5: "0.00",
  6: "0.00",
  7: "0.00",
  8: "0.00",
});
writeSheet(workbook, "Germany Grid Summary", germanyGrid, "GermanyGridSummary", {
  3: "0.00",
  4: "0.00",
  5: "0.00",
  6: "0.00",
  9: "0.00",
});

const parisDailySheet = workbook.worksheets.getItem("Paris Daily");
const parisChart = parisDailySheet.charts.add("line", parisDailySheet.getRange(`A1:D${parisDaily.length}`));
parisChart.title = "Paris Daily Temperature";
parisChart.hasLegend = true;
parisChart.xAxis = { axisType: "textAxis" };
parisChart.yAxis = { numberFormatCode: "0.0" };
parisChart.setPosition("K2", "R18");

const germanyMonthlySheet = workbook.worksheets.getItem("Germany Monthly");
const germanyChart = germanyMonthlySheet.charts.add("bar", germanyMonthlySheet.getRange(`A1:B${germanyMonthly.length}`));
germanyChart.title = "Germany Monthly Mean 10m Wind";
germanyChart.hasLegend = false;
germanyChart.xAxis = { axisType: "textAxis" };
germanyChart.yAxis = { numberFormatCode: "0.0" };
germanyChart.setPosition("G2", "N18");

const dictionary = workbook.worksheets.add("Data Dictionary");
dictionary.showGridLines = false;
dictionary.getRange("A1:C16").values = [
  ["Field", "Meaning", "Unit"],
  ["time_utc", "Observation timestamp", "UTC"],
  ["latitude / longitude", "ERA5 grid coordinate", "degrees"],
  ["t2m_c", "2m air temperature", "C"],
  ["d2m_c", "2m dew point", "C"],
  ["skt_c", "Skin temperature", "C"],
  ["stl1_c", "Soil temperature layer 1", "C"],
  ["msl_hpa", "Mean sea-level pressure", "hPa"],
  ["sp_hpa", "Surface pressure", "hPa"],
  ["tp_mm", "Total precipitation", "mm"],
  ["u10 / v10", "10m wind vector components", "m/s"],
  ["wind10_mps", "10m wind speed from vector magnitude", "m/s"],
  ["wind10_direction_deg", "Meteorological wind direction", "degrees"],
  ["u100 / v100", "100m wind vector components", "m/s"],
  ["swvl1", "Volumetric soil water layer 1", "m3/m3"],
  ["blh", "Boundary layer height", "m"],
];
dictionary.getRange("A1:C1").format = { fill: "#205A6B", font: { bold: true, color: "#FFFFFF" } };
dictionary.getRange("A1:C16").format.borders = { preset: "all", style: "thin", color: "#D9E2E7" };
dictionary.getRange("A:C").format.autofitColumns();

for (const sheetName of [
  "Summary",
  "Paris Daily",
  "Paris Hourly Mean",
  "Germany Daily",
  "Germany Monthly",
  "Data Dictionary",
]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewDir, `${sheetName.replaceAll(" ", "_").toLowerCase()}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "weather_analysis_workbook.xlsx");
await xlsx.save(outputPath);
console.log(outputPath);
