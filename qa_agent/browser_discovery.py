import json
import logging
import re
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from qa_agent.models import (
    DiscoveryResult, DiscoveryStatus, InteractiveElement,
    NavigationAction, NavigationSequence,
)


MAX_SNAPSHOT_CHARS = 8000
MAX_HEADINGS = 8
MAX_LINKS = 10
MAX_BUTTONS = 8
MAX_VISIBLE_TEXT_ELEMENTS = 16
MAX_NAVIGATION_PATHS = 3
MAX_MENU_CANDIDATES = 12
MAX_DIRECT_NAVIGATION_PATHS = 3
MAX_DIRECT_NAVIGATION_CANDIDATES = 12
MAX_TEXT_CHARS = 120
MAX_SELECTOR_CHARS = 180
_EXCLUDED_VISIBLE_TEXT_TAGS = {"script", "style", "meta", "link", "noscript", "template"}

_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]*", re.IGNORECASE)
_SNAPSHOT_SCRIPT = r"""() => {
  const isVisible = (element) => {
    const rect = element.getBoundingClientRect();
        for (let current = element; current && current !== document.documentElement;
                 current = current.parentElement) {
            const style = window.getComputedStyle(current);
            if (style.display === "none" || style.visibility === "hidden" ||
                    Number(style.opacity) === 0) return false;
        }
        return rect.width > 0 && rect.height > 0;
  };
    const verifiedSelector = (element) => {
        const tag = element.tagName.toLowerCase();
        const isExactMatch = (selector) => {
            try {
                const matches = document.querySelectorAll(selector);
                return matches.length === 1 && matches[0] === element;
            } catch (_) {
                return false;
            }
        };
        if (element.id) {
            const idSelector = `#${CSS.escape(element.id)}`;
            if (isExactMatch(idSelector)) return idSelector;
        }
        const name = element.getAttribute("name");
        if (name) {
            const nameSelector = `${tag}[name="${CSS.escape(name)}"]`;
            if (isExactMatch(nameSelector)) return nameSelector;
        }
        const path = [];
        let current = element;
        while (current && current !== document.body) {
            const currentTag = current.tagName.toLowerCase();
            const siblings = Array.from(current.parentElement.children)
                .filter((sibling) => sibling.tagName === current.tagName);
            const position = siblings.indexOf(current) + 1;
            path.unshift(siblings.length > 1 ? `${currentTag}:nth-of-type(${position})` : currentTag);
            current = current.parentElement;
        }
        if (current !== document.body || !path.length) return null;
        const selector = `body > ${path.join(" > ")}`;
        return isExactMatch(selector) ? selector : null;
    };
  const describe = (element) => {
    const tag = element.tagName.toLowerCase();
    const id = element.id || "";
    const name = element.getAttribute("name") || "";
    const role = element.getAttribute("role") || "";
    const ariaLabel = element.getAttribute("aria-label") || "";
    const labelText = element.labels?.[0]?.innerText || "";
    const placeholder = element.getAttribute("placeholder") || "";
    const testId = element.getAttribute("data-testid") ||
        element.getAttribute("data-test") || element.getAttribute("data-qa") || "";
        let selector = id ? `#${CSS.escape(id)}` : tag;
        if (!id && /^H[1-6]$/.test(element.tagName)) {
            const normalizedText = (element.innerText || "").replace(/\s+/g, " ").trim();
            if (normalizedText) {
                const textSelector = `${tag}:has-text(${JSON.stringify(normalizedText)})`;
                const matches = Array.from(document.querySelectorAll(tag)).filter((candidate) =>
                    (candidate.innerText || "").replace(/\s+/g, " ").trim().toLowerCase()
                        .includes(normalizedText.toLowerCase())
                );
                const matchIndex = matches.indexOf(element);
                if (matches.length === 1) {
                    selector = textSelector;
                } else if (matchIndex >= 0) {
                    selector = `:nth-match(${textSelector}, ${matchIndex + 1})`;
                }
            }
        }
    const item = {
      tag,
      selector,
      // Never include a control's current value in discovery or recovery data.
      text: (element.innerText || "").trim().slice(0, 120),
    };
    if (id) item.id = id;
    if (name) item.name = name;
    if (role) item.role = role;
    if (ariaLabel) item.aria_label = ariaLabel;
    if (labelText) item.label = labelText.trim().slice(0, 120);
    if (placeholder) item.placeholder = placeholder.trim().slice(0, 120);
    if (testId) item.test_id = testId.trim().slice(0, 180);
    if (tag === "a" && element.href) item.href = element.href;
    return item;
  };
    const textSelector = (element) => {
        const tag = element.tagName.toLowerCase();
        const isExactMatch = (selector) => {
            try {
                const matches = document.querySelectorAll(selector);
                return matches.length === 1 && matches[0] === element;
            } catch (_) {
                return false;
            }
        };
        if (element.id) {
            const idSelector = `#${CSS.escape(element.id)}`;
            if (isExactMatch(idSelector)) return idSelector;
        }
        const name = element.getAttribute("name");
        if (name) {
            const nameSelector = `${tag}[name="${CSS.escape(name)}"]`;
            if (isExactMatch(nameSelector)) return nameSelector;
        }

        const path = [];
        let current = element;
        while (current && current !== document.body) {
            const currentTag = current.tagName.toLowerCase();
            const siblings = Array.from(current.parentElement.children)
                .filter((sibling) => sibling.tagName === current.tagName);
            const position = siblings.indexOf(current) + 1;
            path.unshift(siblings.length > 1 ? `${currentTag}:nth-of-type(${position})` : currentTag);
            current = current.parentElement;
        }
        if (current !== document.body || !path.length) return null;
        const selector = `body > ${path.join(" > ")}`;
        return isExactMatch(selector) ? selector : null;
    };
    const collectVisibleText = () => {
        const selector = "h1, h2, h3, h4, h5, h6, p, li, td, th, blockquote, pre, figcaption, label, summary, dt, dd, div, span";
        const candidates = Array.from(document.querySelectorAll(selector));
        const hasUsefulText = (element) =>
            isVisible(element) && (element.innerText || "").trim().length > 0;
        const visibleHeadings = Array.from(
            document.querySelectorAll("h1, h2, h3, h4, h5, h6")
        ).filter(isVisible).slice(0, 8);
        const overflowHeadings = candidates.filter((element) =>
            /^H[1-6]$/.test(element.tagName) &&
            !visibleHeadings.includes(element) && hasUsefulText(element)
        );
        const otherCandidates = candidates.filter(
            (element) => !/^H[1-6]$/.test(element.tagName)
        );
        const elements = [];
        for (const element of [...overflowHeadings, ...otherCandidates]) {
            const tag = element.tagName.toLowerCase();
            if (["script", "style", "meta", "link", "noscript", "template"].includes(tag)) continue;
            if (!hasUsefulText(element)) continue;
            if (element.closest("a, button, [role=button]")) continue;
            if (element.parentElement?.closest("h1, h2, h3, h4, h5, h6")) continue;
            if (/^H[1-6]$/.test(element.tagName) && visibleHeadings.includes(element)) continue;

            const generic = tag === "div" || tag === "span";
            if (generic && element.closest("p, li, td, th, blockquote, pre, figcaption, label, summary, dt, dd")) continue;
            if (generic && candidates.some((child) => child !== element && element.contains(child) && hasUsefulText(child))) continue;
            if (!generic && candidates.some((child) => child !== element && element.contains(child) &&
                    !["div", "span"].includes(child.tagName.toLowerCase()) && hasUsefulText(child))) continue;

            const selector = textSelector(element);
            if (!selector) continue;
            elements.push({
                tag,
                selector,
                text: element.innerText.trim().slice(0, 120),
                visible: true,
            });
            if (elements.length >= 16) break;
        }
        return elements;
    };
  const collect = (selector, limit) => {
    const items = [];
    for (const element of document.querySelectorAll(selector)) {
      if (isVisible(element)) items.push(describe(element));
      if (items.length >= limit) break;
    }
    return items;
  };
    const collectInteractive = () => {
        const selector = 'a[href], button, input:not([type=hidden]), select, textarea, [role]';
        return Array.from(document.querySelectorAll(selector)).filter(isVisible).slice(0, 32).map((element) => {
            const selector = verifiedSelector(element);
            if (!selector || selector.length > 180) return null;
            const item = describe(element);
            item.selector = selector;
            const tag = element.tagName.toLowerCase();
            const role = element.getAttribute('role') || '';
            const type = (element.getAttribute('type') || '').toLowerCase();
            const kind = role || (tag === 'input' && ['checkbox', 'radio'].includes(type) ? type : tag);
            item.kind = kind;
            // Keep placeholder separate so recovery can treat it as weaker
            // evidence than an accessible name or an explicit label.
            item.accessible_name = (element.getAttribute('aria-label') || item.label || item.text || '').trim().slice(0, 120);
            const dialog = element.closest('dialog, [role="dialog"], [aria-modal="true"]');
            if (dialog) {
                const dialogId = (dialog.getAttribute('id') || '').trim();
                const dialogLabel = (dialog.getAttribute('aria-label') || '').trim();
                const labelledBy = (dialog.getAttribute('aria-labelledby') || '').trim();
                item.dialog_identity = dialogId ? `id:${dialogId}`
                    : dialogLabel ? `label:${dialogLabel}`
                    : labelledBy ? `labelledby:${labelledBy}`
                    : 'unidentified-dialog';
                item.dialog_identity = item.dialog_identity.slice(0, 120);
            }
            item.visible = true;
            item.enabled = !element.disabled && element.getAttribute('aria-disabled') !== 'true';
            return item;
        }).filter(Boolean);
    };
    const visibleHeadingElements = Array.from(
        document.querySelectorAll("h1, h2, h3, h4, h5, h6")
    ).filter(isVisible).slice(0, 8);
  return {
    url: window.location.href,
    title: document.title,
        headings: visibleHeadingElements.map(describe),
    links: collect("a[href]", 10),
    buttons: collect("button, input[type=button], input[type=submit], [role=button]", 8),
    interactive_elements: collectInteractive(),
    visible_text_elements: collectVisibleText(),
  };
}"""

_NAVIGATION_MENU_SCRIPT = r"""() => {
    const normalize = (value) => (value || "").replace(/\s+/g, " ").trim();
    const isVisible = (element) => {
        const rect = element.getBoundingClientRect();
        for (let current = element; current && current !== document.documentElement;
                 current = current.parentElement) {
            const style = window.getComputedStyle(current);
            if (style.display === "none" || style.visibility === "hidden" ||
                    Number(style.opacity) === 0) return false;
        }
        return rect.width > 0 && rect.height > 0;
    };
    const menuRoot = document.querySelector("#main-menu");
    const uniqueLabel = (value) => {
        const text = normalize(value);
        const half = text.length / 2;
        return Number.isInteger(half) && text.slice(0, half) === text.slice(half)
            ? text.slice(0, half)
            : text;
    };
    const menuButton = document.querySelector("#menu-button-open") ||
        Array.from(document.querySelectorAll("nav button, [role=navigation] button"))
            .find((button) => normalize(button.innerText) === "Meny");
    const tabs = menuRoot
        ? Array.from(menuRoot.querySelectorAll('[role="tab"]')).map((tab) => ({
                text: uniqueLabel(tab.innerText),
                selector: tab.id ? `#${CSS.escape(tab.id)}` : null,
                id: tab.id || null,
                panel_selector: tab.getAttribute("aria-controls")
                    ? `#${CSS.escape(tab.getAttribute("aria-controls"))}`
                    : null,
            }))
        : [];
    const tabLabels = new Map(tabs.map((tab) => [tab.id, tab.text]));
    const items = menuRoot
        ? Array.from(menuRoot.querySelectorAll('a[role="menuitem"][href]'))
                .filter(isVisible)
                .map((element) => {
                    const href = element.getAttribute("href");
                    const selector = `#main-menu a[role="menuitem"][href=${JSON.stringify(href)}]`;
                    const panel = element.closest('[role="tabpanel"]');
                    const tabId = panel ? panel.getAttribute("aria-labelledby") : null;
                    return {
                        text: normalize(
                            element.querySelector(":scope > span")?.innerText || element.innerText
                        ).split(/\s{2,}/)[0].trim(),
                        selector: document.querySelectorAll(selector).length === 1 ? selector : null,
                        href: element.href,
                        tab_text: tabLabels.get(tabId) || "",
                        tab_selector: tabId ? `#${CSS.escape(tabId)}` : null,
                        visible: isVisible(element),
                    };
                })
        : [];

    return {
        menu_button: menuButton
            ? {
                    text: normalize(menuButton.innerText) || menuButton.getAttribute("aria-label") || "Meny",
                    selector: menuButton.id ? `#${CSS.escape(menuButton.id)}` : 'button:has-text("Meny")',
                    visible: isVisible(menuButton),
                }
            : null,
        tabs,
        items,
    };
}"""

_SUBMENU_CANDIDATES_SCRIPT = r"""({parentUrl, scope}) => {
    const normalize = (value) => (value || "").replace(/\s+/g, " ").trim();
    const isVisible = (element) => {
        const rect = element.getBoundingClientRect();
        for (let current = element; current && current !== document.documentElement;
                 current = current.parentElement) {
            const style = window.getComputedStyle(current);
            if (style.display === "none" || style.visibility === "hidden" ||
                    Number(style.opacity) === 0) return false;
        }
        return rect.width > 0 && rect.height > 0;
    };
    const parent = new URL(parentUrl);
    const root = scope === "menu"
        ? document.querySelector("#main-menu")
        : document.querySelector("main, [role=main]") || document.body;
    if (!root) return [];
    const rootSelector = scope === "menu"
        ? "#main-menu"
        : root.tagName.toLowerCase() === "main"
            ? "main"
            : root.getAttribute("role") === "main" ? '[role="main"]' : "body";
    const parentDepth = parent.pathname.split("/").filter(Boolean).length;
    const results = [];
    for (const element of root.querySelectorAll("a[href]")) {
        const url = new URL(element.href);
        const depth = url.pathname.split("/").filter(Boolean).length;
        if (url.origin !== parent.origin ||
                !url.pathname.startsWith(parent.pathname.replace(/\/$/, "") + "/") ||
                depth !== parentDepth + 1) continue;

        const href = element.getAttribute("href");
        const selectorCandidates = element.id
            ? [`#${CSS.escape(element.id)}`]
            : [
                `${rootSelector} li a[href=${JSON.stringify(href)}]`,
                `${rootSelector} a[role="menuitem"][href=${JSON.stringify(href)}]`,
                `${rootSelector} a[href=${JSON.stringify(href)}]`,
            ];
        const selector = selectorCandidates.find(
            (candidate) => document.querySelectorAll(candidate).length === 1
        );
        if (!selector) continue;
        const text = normalize(element.innerText || element.getAttribute("aria-label"));
        if (!text) continue;
        results.push({text, selector, href: element.href, visible: isVisible(element)});
    }
    return results;
}"""

_DISCOVERY_CONTENT_READY_SCRIPT = r"""() => {
    const root = document.querySelector("main, [role=main]");
    if (!root) return true;
    const style = window.getComputedStyle(root);
    const rect = root.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden" &&
        Number(style.opacity) > 0 && rect.width > 0 && rect.height > 0;
}"""

_DISCOVERY_SUBMENU_READY_SCRIPT = r"""({parentUrl}) => {
    const parent = new URL(parentUrl);
    const root = document.querySelector("main, [role=main]");
    if (window.location.origin !== parent.origin ||
            window.location.pathname.replace(/\/$/, "") !== parent.pathname.replace(/\/$/, "") ||
            !root) return false;
    const parentDepth = parent.pathname.split("/").filter(Boolean).length;
    return Array.from(root.querySelectorAll("a[href]")).some((element) => {
        const url = new URL(element.href);
        return url.origin === parent.origin &&
            url.pathname.startsWith(parent.pathname.replace(/\/$/, "") + "/") &&
            url.pathname.split("/").filter(Boolean).length === parentDepth + 1;
    });
}"""

_DIRECT_NAVIGATION_LINKS_SCRIPT = r"""({scope}) => {
    const normalize = (value) => (value || "").replace(/\s+/g, " ").trim();
    const isVisible = (element) => {
        const rect = element.getBoundingClientRect();
        for (let current = element; current && current !== document.documentElement;
                 current = current.parentElement) {
            const style = window.getComputedStyle(current);
            if (style.display === "none" || style.visibility === "hidden" ||
                    Number(style.opacity) === 0) return false;
        }
        return rect.width > 0 && rect.height > 0;
    };
    const root = scope === "root"
        ? document
        : document.querySelector("main, [role=main]");
    if (!root) return [];
    const pageOrigin = window.location.origin === "null"
        ? new URL(document.baseURI).origin
        : window.location.origin;
    const navSelector = scope === "root"
        ? "nav a[href], [role=navigation] a[href]"
        : "nav a[href], [role=navigation] a[href]";
    const selectorFor = (element) => {
        if (element.id) {
            const idSelector = `#${CSS.escape(element.id)}`;
            if (document.querySelectorAll(idSelector).length === 1) return idSelector;
        }
        const href = element.getAttribute("href");
        const quotedHref = JSON.stringify(href);
        const candidates = scope === "root"
            ? [`main nav a[href=${quotedHref}]`, `main [role="navigation"] a[href=${quotedHref}]`,
               `nav a[href=${quotedHref}]`, `[role="navigation"] a[href=${quotedHref}]`,
               `a[href=${quotedHref}]`]
            : [`main nav a[href=${quotedHref}]`, `main [role="navigation"] a[href=${quotedHref}]`,
               `nav a[href=${quotedHref}]`, `[role="navigation"] a[href=${quotedHref}]`];
        return candidates.find((selector) => {
            const matches = document.querySelectorAll(selector);
            return matches.length === 1 && matches[0] === element;
        }) || null;
    };
    const results = [];
    for (const element of root.querySelectorAll(navSelector)) {
        if (element.closest("footer") || !isVisible(element)) continue;
        const href = element.getAttribute("href");
        const resolvedUrl = new URL(href, document.baseURI);
        if (resolvedUrl.origin !== pageOrigin ||
                (resolvedUrl.pathname === window.location.pathname &&
                 resolvedUrl.search === window.location.search)) continue;
        const selector = selectorFor(element);
        const text = normalize(element.getAttribute("aria-label") ||
            element.getAttribute("title") || element.getAttribute("data-icon-name") ||
            element.innerText);
        if (!selector || !text) continue;
        results.push({text, selector, href, resolved_url: resolvedUrl.href});
    }
    return results;
}"""

_DIRECT_DESTINATION_HEADING_SCRIPT = r"""() => {
    const root = document.querySelector("main, [role=main]") || document.body;
    const isVisible = (element) => {
        const rect = element.getBoundingClientRect();
        for (let current = element; current && current !== document.documentElement;
                 current = current.parentElement) {
            const style = window.getComputedStyle(current);
            if (style.display === "none" || style.visibility === "hidden" ||
                    Number(style.opacity) === 0) return false;
        }
        return rect.width > 0 && rect.height > 0;
    };
    for (const element of root.querySelectorAll("h1")) {
        const text = (element.innerText || "").replace(/\s+/g, " ").trim();
        if (!text || !isVisible(element)) continue;
        const selector = element.id &&
            document.querySelectorAll(`#${CSS.escape(element.id)}`).length === 1
                ? `#${CSS.escape(element.id)}`
                : `h1:has-text(${JSON.stringify(text)})`;
        const matches = Array.from(document.querySelectorAll(selector));
        if (matches.length === 1 && matches[0] === element) return {text, selector};
    }
    return null;
}"""


def _discover_navigation_paths(
    page: Any, navigation_menu: Any
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    normalized_menu = _normalize_navigation_menu(navigation_menu)
    if not isinstance(navigation_menu, dict):
        return normalized_menu, []
    menu_button = normalized_menu.get("menu_button")
    if not isinstance(menu_button, dict) or not menu_button.get("selector"):
        return normalized_menu, []

    home_url = page.url
    consent_button = page.locator("#cookie-popup-strictlyNecessary")
    try:
        if consent_button.count():
            consent_button.click(timeout=1500)
    except Exception:
        pass

    tabs = normalized_menu.get("tabs", [])
    menu_items: list[dict[str, str]] = []
    try:
        if not _open_menu_with_single_retry(page, menu_button["selector"]):
            return normalized_menu, []
        for tab in tabs:
            tab_locator = page.locator(tab["selector"])
            try:
                tab_locator.wait_for(state="visible", timeout=2000)
                tab_locator.click()
                panel_selector = tab.get("panel_selector")
                if panel_selector:
                    page.locator(panel_selector).wait_for(
                        state="visible", timeout=2000
                    )
            except Exception:
                continue
            active_menu = page.evaluate(_NAVIGATION_MENU_SCRIPT)
            active_items = active_menu.get("items", []) if isinstance(active_menu, dict) else []
            for item in active_items:
                if not isinstance(item, dict) or not item.get("href") or not item.get("selector"):
                    continue
                menu_items.append(
                    {
                        "text": _bounded_text(item.get("text"), MAX_TEXT_CHARS),
                        "selector": _bounded_text(item["selector"], MAX_SELECTOR_CHARS),
                        "href": item["href"],
                        "tab_text": tab["text"],
                        "tab_selector": tab["selector"],
                    }
                )
    except Exception:
        menu_items = []

    normalized_menu["items"] = menu_items[:MAX_MENU_CANDIDATES]
    base_url = urlsplit(home_url)
    candidates = [
        item
        for item in normalized_menu["items"]
        if urlsplit(item["href"]).netloc == base_url.netloc
    ]

    paths: list[dict[str, str]] = []
    for menu_item in candidates:
        try:
            page.goto(home_url)
            page.wait_for_load_state("load")
            if not _open_menu_with_single_retry(page, menu_button["selector"]):
                continue
            tab_locator = page.locator(menu_item["tab_selector"])
            tab_locator.wait_for(state="visible", timeout=2000)
            tab_locator.click()
            panel_selector = next(
                (
                    tab.get("panel_selector")
                    for tab in tabs
                    if tab.get("selector") == menu_item["tab_selector"]
                ),
                None,
            )
            if panel_selector:
                page.locator(panel_selector).wait_for(
                    state="visible", timeout=2000
                )

            menu_item_locator = page.locator(menu_item["selector"])
            menu_item_locator.wait_for(state="visible", timeout=2000)
            previous_url = page.url
            try:
                with page.expect_navigation(wait_until="commit", timeout=2000):
                    menu_item_locator.click()
            except PlaywrightTimeoutError:
                # Some top-level menu items only expand in place.
                pass
            navigation_occurred = page.url != previous_url
            if navigation_occurred:
                page.wait_for_function(
                    _DISCOVERY_SUBMENU_READY_SCRIPT,
                    arg={"parentUrl": menu_item["href"]},
                    timeout=5000,
                )
                submenu_scope = "main"
            else:
                # The selected tab belongs to the homepage menu, not the
                # destination page; inspect the expanded menu only without nav.
                submenu_scope = "menu"
            submenus = page.evaluate(
                _SUBMENU_CANDIDATES_SCRIPT,
                {"parentUrl": menu_item["href"], "scope": submenu_scope},
            )
        except Exception:
            logging.exception(
                "Navigation path discovery failed for %s",
                menu_item.get("href"),
            )
            continue
        if not isinstance(submenus, list):
            continue

        ordered_submenus = sorted(
            (item for item in submenus if isinstance(item, dict)),
            key=lambda item: (item.get("text", ""), item.get("href", "")),
        )
        for submenu in ordered_submenus:
            try:
                submenu_locator = page.locator(submenu["selector"])
                submenu_locator.wait_for(state="visible", timeout=2000)
                if not submenu_locator.is_visible():
                    continue
                page.goto(submenu["href"])
                page.wait_for_load_state("load")
                page.wait_for_function(
                    _DISCOVERY_CONTENT_READY_SCRIPT, timeout=5000
                )
                destination = page.evaluate(_SNAPSHOT_SCRIPT)
            except Exception:
                continue
            headings = (
                destination.get("headings", [])
                if isinstance(destination, dict)
                else []
            )
            heading = next(
                (
                    item
                    for item in headings
                    if item.get("tag") == "h1"
                    and item.get("selector")
                    and item.get("text")
                ),
                None,
            )
            if heading is None:
                continue

            paths.append(
                {
                    "menu_button_selector": menu_button["selector"],
                    "menu_tab_text": _bounded_text(menu_item.get("tab_text"), 80),
                    "menu_tab_selector": _bounded_text(
                        menu_item.get("tab_selector"), MAX_SELECTOR_CHARS
                    ),
                    "menu_item_text": _bounded_text(
                        menu_item.get("text"), MAX_TEXT_CHARS
                    ),
                    "menu_item_selector": _bounded_text(
                        menu_item["selector"], MAX_SELECTOR_CHARS
                    ),
                    "submenu_text": _bounded_text(
                        submenu.get("text"), MAX_TEXT_CHARS
                    ),
                    "submenu_selector": _bounded_text(
                        submenu["selector"], MAX_SELECTOR_CHARS
                    ),
                    "expected_url": _bounded_text(destination.get("url"), 500),
                    "heading_text": _bounded_text(
                        heading.get("text"), MAX_TEXT_CHARS
                    ),
                    "heading_selector": _bounded_text(
                        heading["selector"], MAX_SELECTOR_CHARS
                    ),
                }
            )
            break
        if len(paths) >= MAX_NAVIGATION_PATHS:
            break

    return normalized_menu, paths


def _open_menu_with_single_retry(page: Any, menu_button_selector: str) -> bool:
    menu_button = page.locator(menu_button_selector)
    menu_button.wait_for(state="visible", timeout=2000)
    main_menu = page.locator("#main-menu")

    menu_button.click()
    if main_menu.is_visible():
        return True

    print("Navigation discovery: menu did not open after first click; retrying.")
    menu_button.click()
    if main_menu.is_visible():
        print("Navigation discovery: menu opened after retry.")
        return True

    print("Navigation discovery: menu did not open after retry; skipping candidate.")
    return False


def _discover_direct_navigation_paths(
    page: Any, root_url: str
) -> list[dict[str, Any]]:
    """Discover replayable two-link paths through ordinary page navigation."""
    try:
        current_url = getattr(page, "url", None)
        if isinstance(current_url, str) and current_url != root_url:
            current_parts = urlsplit(current_url)
            root_parts = urlsplit(root_url)
            same_root = (
                current_parts.scheme == root_parts.scheme
                and current_parts.netloc == root_parts.netloc
                and current_parts.path.rstrip("/") == root_parts.path.rstrip("/")
                and current_parts.query == root_parts.query
            )
            if not same_root:
                page.goto(root_url)
                page.wait_for_load_state("load")
        blocking_dialog = page.locator('[role="dialog"][aria-modal="true"]')
        if blocking_dialog.count() and blocking_dialog.first.is_visible():
            logging.warning(
                "Direct navigation discovery skipped because a modal dialog blocks page interaction."
            )
            return []
        root_links = page.evaluate(
            _DIRECT_NAVIGATION_LINKS_SCRIPT, {"scope": "root"}
        )
    except Exception:
        logging.exception("Direct navigation discovery could not inspect %s", root_url)
        return []
    if not isinstance(root_links, list):
        return []

    root_origin = urlsplit(root_url).netloc
    paths: list[dict[str, Any]] = []
    seen_root_urls: set[str] = set()
    seen_destinations: set[str] = set()

    for root_link in root_links[:MAX_DIRECT_NAVIGATION_CANDIDATES]:
        if not isinstance(root_link, dict):
            continue
        selector = _bounded_text(root_link.get("selector"), MAX_SELECTOR_CHARS)
        href = _bounded_text(root_link.get("href"), 500)
        resolved_url = _bounded_text(root_link.get("resolved_url"), 500)
        text = _bounded_text(root_link.get("text"), MAX_TEXT_CHARS)
        if not all((selector, href, resolved_url, text)):
            continue
        if urlsplit(resolved_url).netloc != root_origin or resolved_url in seen_root_urls:
            continue
        seen_root_urls.add(resolved_url)

        try:
            page.goto(root_url)
            page.wait_for_load_state("load")
            source = page.locator(selector)
            if source.count() != 1 or not source.is_visible():
                continue
            source_url = page.url
            try:
                with page.expect_navigation(wait_until="commit", timeout=5000):
                    source.click(timeout=5000)
            except PlaywrightTimeoutError:
                if page.url == source_url:
                    logging.warning(
                        "Direct navigation source click did not navigate; skipping %s",
                        selector,
                    )
                    continue
            page.wait_for_load_state("load", timeout=10000)
            resolved_root_url = page.url
            if resolved_root_url == source_url or urlsplit(resolved_root_url).netloc != root_origin:
                continue
            child_links = page.evaluate(
                _DIRECT_NAVIGATION_LINKS_SCRIPT, {"scope": "page"}
            )
        except PlaywrightTimeoutError:
            continue
        except Exception:
            logging.exception(
                "Direct navigation source action failed for selector %s", selector
            )
            continue
        if not isinstance(child_links, list):
            continue

        for child_link in child_links[:MAX_DIRECT_NAVIGATION_CANDIDATES]:
            if not isinstance(child_link, dict):
                continue
            child_selector = _bounded_text(
                child_link.get("selector"), MAX_SELECTOR_CHARS
            )
            child_href = _bounded_text(child_link.get("href"), 500)
            child_target_url = _bounded_text(child_link.get("resolved_url"), 500)
            child_text = _bounded_text(child_link.get("text"), MAX_TEXT_CHARS)
            if not all((child_selector, child_href, child_target_url, child_text)):
                continue
            if urlsplit(child_target_url).netloc != root_origin:
                continue
            try:
                page.goto(resolved_root_url)
                page.wait_for_load_state("load")
                child = page.locator(child_selector)
                if child.count() != 1 or not child.is_visible():
                    continue
                before_child_url = page.url
                try:
                    with page.expect_navigation(wait_until="commit", timeout=5000):
                        child.click(timeout=5000)
                except PlaywrightTimeoutError:
                    if page.url == before_child_url:
                        logging.warning(
                            "Direct navigation child click did not navigate; skipping %s",
                            child_selector,
                        )
                        continue
                page.wait_for_load_state("load", timeout=10000)
                expected_url = page.url
                if expected_url == before_child_url or urlsplit(expected_url).netloc != root_origin:
                    continue
                heading = page.evaluate(_DIRECT_DESTINATION_HEADING_SCRIPT)
                if not isinstance(heading, dict):
                    continue
                heading_selector = _bounded_text(
                    heading.get("selector"), MAX_SELECTOR_CHARS
                )
                heading_text = _bounded_text(heading.get("text"), MAX_TEXT_CHARS)
                heading_locator = page.locator(heading_selector)
                if (
                    not heading_selector
                    or not heading_text
                    or heading_locator.count() != 1
                    or not heading_locator.is_visible()
                ):
                    continue
            except PlaywrightTimeoutError:
                continue
            except Exception:
                logging.exception(
                    "Direct navigation child action failed for selector %s",
                    child_selector,
                )
                continue

            if expected_url in seen_destinations:
                continue
            seen_destinations.add(expected_url)
            paths.append(
                {
                    "strategy": "direct_nav",
                    "root_url": root_url,
                    "steps": [
                        {
                            "text": text,
                            "selector": selector,
                            "href": href,
                            "resolved_url": resolved_root_url,
                        },
                        {
                            "text": child_text,
                            "selector": child_selector,
                            "href": child_href,
                            "resolved_url": child_target_url,
                        },
                    ],
                    "expected_url": expected_url,
                    "heading_text": heading_text,
                    "heading_selector": heading_selector,
                }
            )
            if len(paths) >= MAX_DIRECT_NAVIGATION_PATHS:
                return paths

    return paths


def _normalize_navigation_menu(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"menu_button": None, "tabs": [], "items": []}

    button = value.get("menu_button")
    menu_button = None
    if isinstance(button, dict) and button.get("selector"):
        menu_button = {
            "text": _bounded_text(button.get("text"), MAX_TEXT_CHARS),
            "selector": _bounded_text(button.get("selector"), MAX_SELECTOR_CHARS),
        }

    tabs: list[dict[str, str]] = []
    raw_tabs = value.get("tabs")
    if isinstance(raw_tabs, list):
        for tab in raw_tabs[:6]:
            if not isinstance(tab, dict) or not tab.get("selector") or not tab.get("text"):
                continue
            tabs.append(
                {
                    "text": _bounded_text(tab.get("text"), MAX_TEXT_CHARS),
                    "selector": _bounded_text(tab.get("selector"), MAX_SELECTOR_CHARS),
                    "panel_selector": _bounded_text(
                        tab.get("panel_selector"), MAX_SELECTOR_CHARS
                    ),
                }
            )

    items: list[dict[str, str]] = []
    raw_items = value.get("items")
    if isinstance(raw_items, list):
        for item in raw_items[:MAX_MENU_CANDIDATES]:
            if not isinstance(item, dict):
                continue
            normalized = {
                "text": _bounded_text(item.get("text"), MAX_TEXT_CHARS),
                "selector": _bounded_text(item.get("selector"), MAX_SELECTOR_CHARS),
                "href": _bounded_text(item.get("href"), 500),
                "tab_text": _bounded_text(item.get("tab_text"), 80),
                "tab_selector": _bounded_text(
                    item.get("tab_selector"), MAX_SELECTOR_CHARS
                ),
            }
            if all(normalized.values()):
                items.append(normalized)
    return {"menu_button": menu_button, "tabs": tabs, "items": items}


def _normalize_navigation_paths(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    fields = (
        "menu_button_selector",
        "menu_tab_text",
        "menu_tab_selector",
        "menu_item_text",
        "menu_item_selector",
        "submenu_text",
        "submenu_selector",
        "expected_url",
        "heading_text",
        "heading_selector",
    )
    paths: list[dict[str, str]] = []
    for path in value[:MAX_NAVIGATION_PATHS]:
        if not isinstance(path, dict):
            continue
        normalized = {
            field: _bounded_text(
                path.get(field),
                500 if field == "expected_url" else MAX_SELECTOR_CHARS,
            )
            for field in fields
        }
        required = (
            "menu_button_selector",
            "menu_item_selector",
            "submenu_selector",
            "expected_url",
            "heading_text",
            "heading_selector",
        )
        if all(normalized[field] for field in required):
            paths.append(normalized)
    return paths


def _normalize_direct_navigation_paths(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    paths: list[dict[str, Any]] = []
    for path in value[:MAX_DIRECT_NAVIGATION_PATHS]:
        if not isinstance(path, dict) or path.get("strategy") != "direct_nav":
            continue
        raw_steps = path.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != 2:
            continue
        steps: list[dict[str, str]] = []
        for step in raw_steps:
            if not isinstance(step, dict):
                steps = []
                break
            normalized_step = {
                "text": _bounded_text(step.get("text"), MAX_TEXT_CHARS),
                "selector": _bounded_text(step.get("selector"), MAX_SELECTOR_CHARS),
                "href": _bounded_text(step.get("href"), 500),
                "resolved_url": _bounded_text(step.get("resolved_url"), 500),
            }
            if not all(normalized_step.values()):
                steps = []
                break
            steps.append(normalized_step)
        normalized = {
            "strategy": "direct_nav",
            "root_url": _bounded_text(path.get("root_url"), 500),
            "steps": steps,
            "expected_url": _bounded_text(path.get("expected_url"), 500),
            "heading_text": _bounded_text(path.get("heading_text"), MAX_TEXT_CHARS),
            "heading_selector": _bounded_text(
                path.get("heading_selector"), MAX_SELECTOR_CHARS
            ),
        }
        if (
            len(steps) == 2
            and all(normalized[key] for key in (
                "root_url", "expected_url", "heading_text", "heading_selector"
            ))
        ):
            paths.append(normalized)
    return paths


def extract_target_url(task: str) -> str:
    match = _URL_PATTERN.search(task)
    if match is None:
        raise ValueError("The QA task must include an http:// or https:// URL.")

    url = match.group(0).rstrip(".,!?;:")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"The task contains an invalid target URL: {url!r}.")
    return url


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "")[:limit]


def _normalize_items(value: Any, limit: int, include_href: bool = False) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    items: list[dict[str, str]] = []
    for entry in value[:limit]:
        if not isinstance(entry, dict):
            continue
        item = {
            key: _bounded_text(
                entry.get(key),
                MAX_TEXT_CHARS if key == "text" else MAX_SELECTOR_CHARS,
            )
            for key in ("tag", "selector", "id", "name", "role", "aria_label", "text")
            if entry.get(key)
        }
        if include_href and entry.get("href"):
            item["href"] = _bounded_text(entry["href"], 300)
        if item:
            items.append(item)
    return items


def _normalize_visible_text_elements(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    items: list[dict[str, str]] = []
    for entry in value[:MAX_VISIBLE_TEXT_ELEMENTS]:
        if not isinstance(entry, dict) or entry.get("visible") is not True:
            continue
        tag = _bounded_text(entry.get("tag"), MAX_SELECTOR_CHARS).lower()
        text = _bounded_text(entry.get("text"), MAX_TEXT_CHARS).strip()
        selector = _bounded_text(entry.get("selector"), MAX_SELECTOR_CHARS)
        if tag in _EXCLUDED_VISIBLE_TEXT_TAGS or not text or not selector:
            continue
        items.append({"tag": tag, "selector": selector, "text": text})
    return items


def _normalize_interactive_elements(value: Any) -> list[dict[str, Any]]:
    """Keep only bounded, replay-relevant identity and state information."""
    if not isinstance(value, list):
        return []
    fields = (
        "kind", "selector", "text", "accessible_name", "label", "placeholder",
        "test_id", "tag", "role", "id", "name", "href", "dialog_identity",
    )
    result: list[dict[str, Any]] = []
    for entry in value[:32]:
        if not isinstance(entry, dict) or not entry.get("selector"):
            continue
        item = {
            key: _bounded_text(
                entry.get(key),
                500 if key == "href" else 180 if key in {"id", "name", "test_id"}
                else MAX_TEXT_CHARS if key in {"text", "accessible_name", "label", "placeholder"}
                else 120 if key == "dialog_identity"
                else MAX_SELECTOR_CHARS,
            )
            for key in fields if entry.get(key)
        }
        item["visible"] = entry.get("visible") is True
        item["enabled"] = entry.get("enabled") is not False
        if item.get("kind"):
            result.append(item)
    return result


def _build_snapshot(page_data: Any) -> str:
    if not isinstance(page_data, dict):
        raise ValueError("Browser discovery did not return a page snapshot.")

    snapshot: dict[str, Any] = {
        "url": _bounded_text(page_data.get("url"), 500),
        "title": _bounded_text(page_data.get("title"), 200),
        "headings": _normalize_items(page_data.get("headings"), MAX_HEADINGS),
        "links": _normalize_items(page_data.get("links"), MAX_LINKS, include_href=True),
        "buttons": _normalize_items(page_data.get("buttons"), MAX_BUTTONS),
        "interactive_elements": _normalize_interactive_elements(
            page_data.get("interactive_elements")
        ),
        "navigation_menu": _normalize_navigation_menu(
            page_data.get("navigation_menu")
        ),
        "navigation_paths": _normalize_navigation_paths(
            page_data.get("navigation_paths")
        ),
        "direct_navigation_paths": _normalize_direct_navigation_paths(
            page_data.get("direct_navigation_paths")
        ),
        "visible_text_elements": _normalize_visible_text_elements(
            page_data.get("visible_text_elements")
        ),
    }

    serialized = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    truncated = False
    for collection in (
        "buttons",
        "interactive_elements",
        "links",
        "visible_text_elements",
        "headings",
        "direct_navigation_paths",
        "navigation_paths",
    ):
        while len(serialized) > MAX_SNAPSHOT_CHARS and snapshot[collection]:
            snapshot[collection].pop()
            truncated = True
            serialized = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    while (
        len(serialized) > MAX_SNAPSHOT_CHARS
        and snapshot["navigation_menu"]["items"]
    ):
        snapshot["navigation_menu"]["items"].pop()
        truncated = True
        serialized = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    if truncated:
        snapshot["truncated"] = True
        serialized = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        for collection in (
            "buttons",
            "interactive_elements",
            "links",
            "visible_text_elements",
            "headings",
            "direct_navigation_paths",
            "navigation_paths",
        ):
            while len(serialized) > MAX_SNAPSHOT_CHARS and snapshot[collection]:
                snapshot[collection].pop()
                serialized = json.dumps(
                    snapshot, ensure_ascii=False, separators=(",", ":")
                )
        while (
            len(serialized) > MAX_SNAPSHOT_CHARS
            and snapshot["navigation_menu"]["items"]
        ):
            snapshot["navigation_menu"]["items"].pop()
            serialized = json.dumps(
                snapshot, ensure_ascii=False, separators=(",", ":")
            )
    return serialized


def capture_page_snapshot(url: str) -> str:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        try:
            page = browser.new_page()
            page.goto(url)
            page.wait_for_load_state("load")
            page_data = page.evaluate(_SNAPSHOT_SCRIPT)
            navigation_menu = page.evaluate(_NAVIGATION_MENU_SCRIPT)
            navigation_menu, navigation_paths = _discover_navigation_paths(
                page, navigation_menu
            )
            direct_navigation_paths = (
                []
                if navigation_paths
                else _discover_direct_navigation_paths(page, url)
            )
            page_data["navigation_menu"] = navigation_menu
            page_data["navigation_paths"] = navigation_paths
            page_data["direct_navigation_paths"] = direct_navigation_paths
            return _build_snapshot(page_data)
        finally:
            browser.close()


def capture_discovery_result(url: str) -> DiscoveryResult:
    """Return typed discovery data while retaining the legacy snapshot API."""
    try:
        snapshot = json.loads(capture_page_snapshot(url))
    except Exception as exc:
        return DiscoveryResult(
            status=DiscoveryStatus.FAILED,
            url=url,
            warnings=[f"{type(exc).__name__}: {exc}"],
            strategies_used=["menu_navigation", "direct_navigation"],
        )

    navigation_paths = snapshot.get("navigation_paths", [])
    direct_navigation_paths = snapshot.get("direct_navigation_paths", [])
    interactive_elements = [InteractiveElement.model_validate(item) for item in snapshot.get("interactive_elements", [])]
    by_selector = {item.selector: item for item in interactive_elements}
    navigation: list[NavigationSequence] = []
    for path in navigation_paths:
        selectors = [path.get(key) for key in (
            "menu_button_selector", "menu_tab_selector", "menu_item_selector", "submenu_selector"
        ) if path.get(key)]
        elements = [by_selector.get(selector) or InteractiveElement(
            kind="navigation_control", selector=selector,
            text=path.get("submenu_text", "") if selector == path.get("submenu_selector") else "",
            accessible_name=path.get("submenu_text", "") if selector == path.get("submenu_selector") else "",
        ) for selector in selectors]
        navigation.append(NavigationSequence(
            actions=[NavigationAction(element=element) for element in elements],
            destination_url=path.get("expected_url", ""),
        ))
    for path in direct_navigation_paths:
        elements = [by_selector.get(step.get("selector")) or InteractiveElement(
            kind="link", selector=step["selector"], text=step.get("text", ""),
            accessible_name=step.get("text", ""), tag="a", href=step.get("href", ""),
        ) for step in path["steps"]]
        navigation.append(NavigationSequence(
            actions=[NavigationAction(element=element) for element in elements],
            destination_url=path.get("expected_url", ""),
        ))
    strategies_used = ["menu_navigation"]
    if not navigation_paths:
        strategies_used.append("direct_navigation")

    has_paths = bool(navigation_paths or direct_navigation_paths)
    warnings = [] if has_paths else ["No navigation paths were discovered."]
    return DiscoveryResult(
        status=DiscoveryStatus.SUCCESS if has_paths else DiscoveryStatus.PARTIAL,
        url=str(snapshot.get("url") or url),
        title=str(snapshot.get("title") or ""),
        snapshot=snapshot,
        navigation_paths=navigation_paths,
        direct_navigation_paths=direct_navigation_paths,
        interactive_elements=interactive_elements,
        navigation=navigation,
        warnings=warnings,
        strategies_used=strategies_used,
    )
