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

  function makeMetricTable(data) {
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
    return '<table class="data-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>' + rows + "</tbody></table>";
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
        makeMetricTable(data) +
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

  Promise.all([
    fetchJson("assets/data/ours_results.json"),
    fetchJson("assets/data/reproduced_sota_results.json"),
    fetchJson("assets/qualitative/manifest.json"),
    fetchJson("assets/data/ours_efficiency.json")
  ]).then(function (payloads) {
    renderOursResults(payloads[0]);
    renderComparison(payloads[0], payloads[1]);
    renderQualitative(payloads[2]);
    renderEfficiency(payloads[3]);
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
