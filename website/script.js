(function () {
  function byId(id) {
    return document.getElementById(id);
  }

  function safeText(value) {
    if (value === null || value === undefined || value === "") {
      return "TBD";
    }
    if (typeof value === "number") {
      return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
    }
    return String(value);
  }

  function toNumeric(value) {
    if (value === null || value === undefined || value === "" || value === "TBD") {
      return null;
    }
    if (typeof value === "number") {
      return Number.isFinite(value) ? value : null;
    }
    var parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function safePercent(value) {
    var numeric = toNumeric(value);
    if (numeric === null) {
      return "TBD";
    }
    return (numeric * 100).toFixed(2);
  }

  function safePercentCell(value) {
    var numeric = toNumeric(value);
    if (numeric === null) {
      return "TBD";
    }
    return numeric.toFixed(2);
  }

  function safeThreshold(value) {
    var numeric = toNumeric(value);
    if (numeric === null) {
      return "TBD";
    }
    return numeric.toFixed(2);
  }

  function renderSourceBadge(sourceKind) {
    var cls = "badge-missing";
    var label = "Pending";
    if (sourceKind === "eval_metrics_json" || sourceKind === "test_metrics_json" || sourceKind === "eval_metrics_csv" || sourceKind === "test_metrics_csv" || sourceKind === "validation_csv" || sourceKind === "provided_verified_log") {
      cls = "badge-reproduced";
      label = "Reproduced";
    } else if (sourceKind === "literature") {
      cls = "badge-literature";
      label = "Literature";
    } else if (sourceKind === "manual_needed") {
      cls = "badge-unverified";
      label = "Manual needed";
    }
    return '<span class="badge ' + cls + '">' + label + "</span>";
  }

  function renderStatusBadge(status) {
    var cls = "badge-missing";
    var label = status || "TBD";
    if (status === "OK") {
      cls = "badge-reproduced";
    } else if (status === "MANUAL_REQUIRED") {
      cls = "badge-unverified";
    } else if (status === "MISSING_CHECKPOINT" || status === "FAILED" || status === "ADAPTER_FAILED") {
      cls = "badge-missing";
    }
    return '<span class="badge ' + cls + '">' + label + "</span>";
  }

  function makeMetricSummary(dataset, data) {
    if (dataset === "SECOND") {
      return "";
    }
    var protocolKeys = [
      ["Pre", safePercent(data.Precision_1)],
      ["Rec", safePercent(data.Recall_1)],
      ["F1", safePercent(data.F1_1)],
      ["IoU", safePercent(data.IoU_1)],
      ["OA", safePercent(data.OA)],
      ["Threshold", safeThreshold(data.threshold)]
    ];
    var hasAnyMetric = protocolKeys.some(function (item) {
      return item[1] !== "TBD";
    });
    if (!hasAnyMetric) {
      return '<p class="status-note">Pending evaluation</p>';
    }
    var cards = protocolKeys.map(function (item) {
      return '<article class="metric-card"><span class="metric-card-label">' + item[0] + '</span><strong class="metric-card-value">' + item[1] + '</strong></article>';
    }).join("");
    return '<div class="metric-card-grid">' + cards + "</div>";
  }

  function makeMetricTable(dataset, data) {
    var metrics = [
      "mF1", "F1_1", "F1_0", "mIoU", "IoU_1", "IoU_0",
      "Precision_1", "Recall_1", "OA", "Boundary F1", "Edge IoU", "threshold"
    ];
    var rows = metrics.map(function (metric) {
      return "<tr><td>" + metric + "</td><td>" + safeText(data[metric]) + "</td></tr>";
    }).join("");
    rows += "<tr><td>Source</td><td>" + renderSourceBadge(data.source_kind) + "</td></tr>";
    rows += "<tr><td>Source file</td><td>" + safeText(data.source_file) + "</td></tr>";
    rows += "<tr><td>Run directory</td><td>" + safeText(data.run_directory) + "</td></tr>";
    return makeMetricSummary(dataset, data) +
      '<div class="table-scroll"><table class="data-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>' + rows + "</tbody></table></div>";
  }

  function renderOursResults(payload) {
    if (!payload || !payload.results) return;
    var container = byId("ours-results-panels");
    if (!container) return;
    var datasets = ["LEVIR-CD", "WHU-CD", "DSIFN-CD", "SECOND"];
    container.innerHTML = datasets.map(function (dataset, index) {
      var data = payload.results[dataset] || { source_kind: "missing" };
      return '<section class="surface-card tab-panel' + (index === 0 ? " is-active" : "") + '" id="ours-' + dataset + '">' +
        "<h3>" + dataset + "</h3>" +
        makeMetricTable(dataset, data) +
        "</section>";
    }).join("");

    if (payload.summary_cards) {
      byId("summary-levir-f1").textContent = safeText(payload.summary_cards.levir_f1_1);
      byId("summary-whu-f1").textContent = safeText(payload.summary_cards.best_whu_f1_1);
      byId("summary-whu-boundary").textContent = safeText(payload.summary_cards.best_whu_boundary_f1);
      if (payload.summary_cards.params_millions !== null && payload.summary_cards.params_millions !== undefined) {
        byId("summary-params").textContent = safeText(payload.summary_cards.params_millions) + "M";
      }
    }
  }

  function renderComparison(oursPayload, reproducedPayload) {
    var container = byId("comparison-panels");
    if (!container) return;
    var datasets = ["LEVIR-CD", "WHU-CD", "DSIFN-CD", "SECOND"];
    var reproducedRecords = (reproducedPayload && reproducedPayload.records) ? reproducedPayload.records : [];
    var panels = datasets.map(function (dataset, index) {
      var rows = [];
      var modelOrder = ["MambaRefine-CD", "ChangeFormer", "BIT", "SNUNet", "STANet", "Mamba-CD"];

      function findRow(model) {
        return reproducedRecords.find(function (record) {
          return record.Model === model && record.Dataset === dataset;
        });
      }

      if (dataset === "SECOND") {
        modelOrder.forEach(function (model) {
          var row = findRow(model) || {};
          rows.push([
            model,
            safeText(row.OA),
            safeText(row.Fscd),
            safeText(row.mIoU),
            safeText(row.SeK),
            renderStatusBadge(row.Status)
          ]);
        });
      } else {
        modelOrder.forEach(function (model) {
          var row = findRow(model) || {};
          var typeLabel = row.Status === "OK" ? "Reproduced" : (row.Status || "Pending");
          rows.push([
            model,
            typeLabel,
            safeText(row.Params_M),
            safeText(row.FLOPs_G),
            safeText(row.F1_1),
            safeText(row.IoU_1),
            safeText(row.OA),
            safeText(row.Boundary_F1),
            renderStatusBadge(row.Status)
          ]);
        });
      }

      var tableHeader;
      if (dataset === "SECOND") {
        tableHeader = "<tr><th>Model</th><th>OA</th><th>Fscd</th><th>mIoU</th><th>SeK</th><th>Source</th></tr>";
      } else {
        tableHeader = "<tr><th>Model</th><th>Type</th><th>Params (M)</th><th>FLOPs</th><th>F1_1</th><th>IoU_1</th><th>OA</th><th>Boundary F1</th><th>Source</th></tr>";
      }
      var tableRows = rows.map(function (row) {
        return "<tr>" + row.map(function (cell) { return "<td>" + cell + "</td>"; }).join("") + "</tr>";
      }).join("");

      return '<section class="surface-card tab-panel' + (index === 0 ? " is-active" : "") + '" id="compare-' + dataset + '">' +
        "<h3>" + dataset + "</h3>" +
        '<table class="data-table"><thead>' + tableHeader + '</thead><tbody>' + tableRows + "</tbody></table>" +
        "</section>";
    });

    container.innerHTML = panels.join("");
  }

  function distinctDescending(values) {
    var out = [];
    values.slice().sort(function (a, b) { return b - a; }).forEach(function (value) {
      if (!out.some(function (item) { return Math.abs(item - value) < 1e-9; })) {
        out.push(value);
      }
    });
    return out;
  }

  function metricRankMap(rows, metricKey) {
    var values = rows.map(function (row) { return toNumeric(row[metricKey]); }).filter(function (value) {
      return value !== null;
    });
    var unique = distinctDescending(values);
    return {
      best: unique.length > 0 ? unique[0] : null,
      second: unique.length > 1 ? unique[1] : null
    };
  }

  function protocolCellClass(value, rankInfo) {
    var numeric = toNumeric(value);
    if (numeric === null) {
      return "";
    }
    if (rankInfo.best !== null && Math.abs(numeric - rankInfo.best) < 1e-9) {
      return "metric-best";
    }
    if (rankInfo.second !== null && Math.abs(numeric - rankInfo.second) < 1e-9) {
      return "metric-second";
    }
    return "";
  }

  function renderProtocolComparison(payload) {
    var container = byId("mambacd-protocol-panels");
    if (!container || !payload || !payload.datasets) return;
    var datasets = ["LEVIR-CD", "WHU-CD", "DSIFN-CD"];
    container.innerHTML = datasets.map(function (dataset, index) {
      var rows = payload.datasets[dataset] || [];
      var preRanks = metricRankMap(rows, "Pre (%)");
      var recRanks = metricRankMap(rows, "Rec (%)");
      var f1Ranks = metricRankMap(rows, "F1 (%)");
      var iouRanks = metricRankMap(rows, "IoU (%)");
      var oaRanks = metricRankMap(rows, "OA (%)");
      var tableRows = rows.map(function (row) {
        var rowClass = "";
        if (row.Method === "MambaRefine-CD") {
          rowClass = " protocol-row-ours";
        } else if (row.Method === "Mamba-CD") {
          rowClass = " protocol-row-mambacd";
        }
        return '<tr class="' + rowClass.trim() + '">' +
          "<td>" + safeText(row.Method) + "</td>" +
          "<td>" + safeText(row.Year) + "</td>" +
          '<td class="' + protocolCellClass(row["Pre (%)"], preRanks) + '">' + safePercentCell(row["Pre (%)"]) + "</td>" +
          '<td class="' + protocolCellClass(row["Rec (%)"], recRanks) + '">' + safePercentCell(row["Rec (%)"]) + "</td>" +
          '<td class="' + protocolCellClass(row["F1 (%)"], f1Ranks) + '">' + safePercentCell(row["F1 (%)"]) + "</td>" +
          '<td class="' + protocolCellClass(row["IoU (%)"], iouRanks) + '">' + safePercentCell(row["IoU (%)"]) + "</td>" +
          '<td class="' + protocolCellClass(row["OA (%)"], oaRanks) + '">' + safePercentCell(row["OA (%)"]) + "</td>" +
          "<td>" + safeText(row.Source) + "</td>" +
          "</tr>";
      }).join("");
      return '<section class="surface-card tab-panel' + (index === 0 ? " is-active" : "") + '" id="protocol-' + dataset + '">' +
        "<h3>" + dataset + "</h3>" +
        '<div class="table-scroll"><table class="data-table"><thead><tr><th>Method</th><th>Year</th><th>Pre (%)</th><th>Rec (%)</th><th>F1 (%)</th><th>IoU (%)</th><th>OA (%)</th><th>Source</th></tr></thead><tbody>' + tableRows + "</tbody></table></div>" +
        "</section>";
    }).join("");
  }

  function renderProtocolNotes(payload) {
    var container = byId("mambacd-protocol-notes");
    if (!container || !payload || !payload.datasets) return;
    var dsifnRows = payload.datasets["DSIFN-CD"] || [];
    var oursDsifn = dsifnRows.find(function (row) { return row.Method === "MambaRefine-CD"; }) || {};
    var dsifnText = toNumeric(oursDsifn["F1 (%)"]) === null
      ? "DSIFN-CD is pending until a local protocol evaluation result is available."
      : "DSIFN-CD local protocol metrics were extracted from local evaluation logs.";
    container.innerHTML = [
      "<li>WHU-CD result is competitive and close to Mamba-CD.</li>",
      "<li>LEVIR-CD is below Mamba-CD in change F1/IoU.</li>",
      "<li>" + dsifnText + "</li>"
    ].join("");
  }

  function renderQualitative(payload) {
    var container = byId("qualitative-grid");
    if (!container) return;
    var items = (payload && payload.items) ? payload.items : [];
    container.innerHTML = items.map(function (item) {
      var dataset = item.dataset || "Unknown";
      if (item.file) {
        return '<article class="surface-card qual-card" data-dataset="' + dataset + '">' +
          '<img src="' + item.file + '" alt="' + dataset + ' qualitative result">' +
          "<h3>" + dataset + "</h3>" +
          "<p>" + safeText(item.caption) + "</p>" +
          "</article>";
      }
      return '<article class="surface-card qual-card placeholder-card" data-dataset="' + dataset + '">' +
        "<h3>" + dataset + "</h3>" +
        "<p>Qualitative result pending. Run scripts/collect_website_qualitative.py after evaluation.</p>" +
        "</article>";
    }).join("");
  }

  function parseCsv(text) {
    if (!text) return [];
    var rows = [];
    var row = [];
    var cell = "";
    var inQuotes = false;
    for (var i = 0; i < text.length; i += 1) {
      var ch = text[i];
      var next = text[i + 1];
      if (ch === '"' && inQuotes && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = !inQuotes;
      } else if (ch === "," && !inQuotes) {
        row.push(cell);
        cell = "";
      } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
        if (ch === "\r" && next === "\n") i += 1;
        row.push(cell);
        if (row.some(function (item) { return item !== ""; })) rows.push(row);
        row = [];
        cell = "";
      } else {
        cell += ch;
      }
    }
    row.push(cell);
    if (row.some(function (item) { return item !== ""; })) rows.push(row);
    if (rows.length < 2) return [];
    var header = rows[0];
    return rows.slice(1).map(function (values) {
      var record = {};
      header.forEach(function (key, index) {
        record[key] = values[index] || "";
      });
      return record;
    });
  }

  function renderAblation(csvText) {
    var tableContainer = byId("ablation-table-container");
    var highlights = byId("ablation-highlights");
    if (!tableContainer || !highlights) return;

    var rows = parseCsv(csvText).filter(function (row) {
      return row.experiment && row.F1_1 !== "";
    });
    if (!rows.length) {
      tableContainer.innerHTML = '<p class="status-note">Ablation table pending. Run scripts/extract_ablation_results.py after experiments finish.</p>';
      highlights.innerHTML = '<p class="status-note">No completed ablation summary is available yet.</p>';
      return;
    }

    var datasets = [];
    rows.forEach(function (row) {
      if (datasets.indexOf(row.dataset) === -1) {
        datasets.push(row.dataset);
      }
    });

    tableContainer.innerHTML = datasets.map(function (dataset) {
      var datasetRows = rows.filter(function (row) { return row.dataset === dataset; });
      var tableRows = datasetRows.map(function (row) {
        return "<tr>" +
          "<td>" + safeText(row.method || row.experiment) + "</td>" +
          "<td>" + safeText(row.F1_1) + "</td>" +
          "<td>" + safeText(row.IoU_1) + "</td>" +
          "<td>" + safeText(row.OA) + "</td>" +
          "<td>" + (row.experiment === "baseline" ? "—" : safeText(row.delta_F1)) + "</td>" +
          "</tr>";
      }).join("");
      return '<section class="ablation-dataset-block"><h4>' + dataset + '</h4><table class="data-table"><thead><tr><th>Method</th><th>F1</th><th>IoU</th><th>OA</th><th>ΔF1</th></tr></thead><tbody>' + tableRows + '</tbody></table></section>';
    }).join("");

    var highlightHtml = [];
    datasets.forEach(function (dataset) {
      var datasetRows = rows.filter(function (row) {
        return row.dataset === dataset && row.experiment !== "baseline" && toNumeric(row.delta_F1) !== null;
      });
      datasetRows.sort(function (a, b) {
        return toNumeric(a.delta_F1) - toNumeric(b.delta_F1);
      });
      if (!datasetRows.length) {
        return;
      }
      var biggest = datasetRows[0];
      var least = datasetRows[datasetRows.length - 1];
      highlightHtml.push('<article class="ablation-highlight"><span>' + dataset + ' most important module</span><strong>' + safeText(biggest.method || biggest.experiment) + '</strong></article>');
      highlightHtml.push('<article class="ablation-highlight"><span>' + dataset + ' least important module</span><strong>' + safeText(least.method || least.experiment) + '</strong></article>');
    });
    highlights.innerHTML = highlightHtml.join("");
  }

  function renderEfficiency(payload) {
    var tbody = document.querySelector("#efficiency-table tbody");
    var status = byId("efficiency-status");
    if (!tbody || !status) return;
    var metrics = payload && payload.metrics ? payload.metrics : {};
    var rows = [
      ["Device", metrics.device],
      ["Variant", metrics.variant],
      ["Decoder", metrics.decoder],
      ["Total params", metrics.total_params_millions ? metrics.total_params_millions + "M" : "TBD"],
      ["Trainable params", metrics.trainable_params_millions ? metrics.trainable_params_millions + "M" : "TBD"],
      ["Backbone params", metrics.backbone_params_millions ? metrics.backbone_params_millions + "M" : "TBD"],
      ["D-RBI params", metrics.drbi_params_millions ? metrics.drbi_params_millions + "M" : "TBD"],
      ["Decoder params", metrics.decoder_params_millions ? metrics.decoder_params_millions + "M" : "TBD"],
      ["FLOPs", metrics.flops_gmac !== undefined ? safeText(metrics.flops_gmac) : "TBD"],
      ["Peak forward memory", metrics.peak_forward_memory_mb !== undefined ? safeText(metrics.peak_forward_memory_mb) : "TBD"],
      ["Peak train-step memory", metrics.peak_train_step_memory_mb !== undefined ? safeText(metrics.peak_train_step_memory_mb) : "TBD"],
      ["FPS", metrics.fps !== undefined ? safeText(metrics.fps) : "TBD"]
    ];
    tbody.innerHTML = rows.map(function (row) {
      return "<tr><td>" + row[0] + "</td><td>" + safeText(row[1]) + "</td></tr>";
    }).join("");
    if (metrics.total_params_millions !== undefined && metrics.total_params_millions !== null) {
      byId("summary-params").textContent = safeText(metrics.total_params_millions) + "M";
    }
    status.textContent = metrics.flops_reason ? "FLOP status: " + metrics.flops_reason : "Efficiency profile loaded from local script output.";
  }

  function initTabs() {
    document.querySelectorAll("[data-tab-group]").forEach(function (group) {
      var buttons = group.querySelectorAll(".tab-button");
      buttons.forEach(function (button) {
        button.addEventListener("click", function () {
          var target = button.getAttribute("data-tab-target");
          buttons.forEach(function (item) { item.classList.remove("is-active"); });
          button.classList.add("is-active");
          document.querySelectorAll("#" + CSS.escape(target) + ", #" + CSS.escape(target.replace(" ", ""))).forEach(function () {});
          var prefix = target.split("-")[0];
          document.querySelectorAll('[id^="' + prefix + '-"]').forEach(function (panel) {
            panel.classList.remove("is-active");
          });
          var actual = document.getElementById(target) || document.getElementById(target.replace(" ", ""));
          if (actual) {
            actual.classList.add("is-active");
            if (window.MathJax && window.MathJax.typesetPromise) {
              window.MathJax.typesetPromise([actual]);
            }
          }
        });
      });
    });
  }

  function initQualitativeFilter() {
    var buttons = document.querySelectorAll(".filter-button");
    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        var filter = button.getAttribute("data-filter");
        buttons.forEach(function (item) { item.classList.remove("is-active"); });
        button.classList.add("is-active");
        document.querySelectorAll(".qual-card").forEach(function (card) {
          var dataset = card.getAttribute("data-dataset");
          card.style.display = (filter === "All" || filter === dataset) ? "" : "none";
        });
      });
    });
  }

  function initDisclosure() {
    document.querySelectorAll(".disclosure-toggle").forEach(function (button) {
      button.addEventListener("click", function () {
        var targetId = button.getAttribute("data-target");
        var target = byId(targetId);
        if (!target) return;
        var isOpen = target.classList.toggle("is-open");
        button.setAttribute("aria-expanded", isOpen ? "true" : "false");
        if (window.MathJax && window.MathJax.typesetPromise) {
          window.MathJax.typesetPromise([target]);
        }
      });
    });
  }

  function initNav() {
    var toggle = document.querySelector(".nav-toggle");
    var links = byId("nav-links");
    if (!toggle || !links) return;
    toggle.addEventListener("click", function () {
      var open = links.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    links.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        links.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  function initProgress() {
    var bar = byId("progress-bar");
    if (!bar) return;
    function update() {
      var top = window.scrollY || document.documentElement.scrollTop || 0;
      var height = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = (height > 0 ? (top / height) * 100 : 0) + "%";
    }
    window.addEventListener("scroll", update, { passive: true });
    update();
  }

  function fetchJson(path) {
    return fetch(path).then(function (response) {
      if (!response.ok) throw new Error("Failed to load " + path);
      return response.json();
    }).catch(function () {
      return null;
    });
  }

  function fetchText(path) {
    return fetch(path).then(function (response) {
      if (!response.ok) throw new Error("Failed to load " + path);
      return response.text();
    }).catch(function () {
      return "";
    });
  }

  Promise.all([
    fetchJson("assets/data/ours_results.json"),
    fetchJson("assets/data/reproduced_sota_results.json"),
    fetchJson("assets/qualitative/manifest.json"),
    fetchJson("assets/data/ours_efficiency.json"),
    fetchJson("assets/data/mambacd_paper_comparison.json"),
    fetchText("assets/tables/ablation_summary.csv")
  ]).then(function (payloads) {
    renderOursResults(payloads[0]);
    renderComparison(payloads[0], payloads[1]);
    renderQualitative(payloads[2]);
    renderEfficiency(payloads[3]);
    renderProtocolComparison(payloads[4]);
    renderProtocolNotes(payloads[4]);
    renderAblation(payloads[5]);
    initTabs();
    initQualitativeFilter();
    initDisclosure();
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  });

  initNav();
  initProgress();
}());
