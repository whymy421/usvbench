# GitHub Collaboration Guide

> How we work together on the USVBench repo. Read this once before your first commit.

Repo: **https://github.com/whymy421/usvbench** (private)

---

## 0. Getting access

1. Send me (Yutong) your **GitHub username**.
2. I'll add you as a **collaborator** (Settings → Collaborators).
3. You'll get an email invite — accept it. Now you can push branches.

You'll also be added to the **wandb** project so your training runs show up alongside mine.

---

## 1. One-time setup

```bash
# Clone into your home dir (so ~/usvbench/assets resolves automatically)
git clone https://github.com/whymy421/usvbench.git ~/usvbench
cd ~/usvbench

# Tell git who you are (use your GitHub email)
git config user.name  "Your Name"
git config user.email "you@example.com"
```

If `git clone` asks for a password, use a **Personal Access Token** (GitHub no longer
accepts account passwords): GitHub → Settings → Developer settings → Personal access
tokens → Fine-grained token → give it `repo` access to `usvbench`. Paste the token as
the password.

---

## 2. The daily workflow (branch → commit → push → PR)

**Never commit directly to `main`.** Always work on your own branch.

```bash
# 1. Start from an up-to-date main
git checkout main
git pull

# 2. Make a branch for your task (name it clearly)
git checkout -b arif/catamaran-patrol

# 3. Do your work, then stage + commit in small logical chunks
git add tasks/catamaran_patrol/
git commit -m "catamaran: add task env + cfg with physics constants"

# 4. Push your branch to GitHub
git push -u origin arif/catamaran-patrol
```

Then open a **Pull Request** on github.com (it'll show a "Compare & pull request"
button after you push). I review it, request changes if needed, and merge into `main`.

---

## 3. Commit message style

Short, present-tense, prefixed with the area:

```
catamaran: add waypoint-sequence reward
docs: fix install path in catamaran STARTER
fix: boat sinks when mass > 300 (clamp rov_volume)
```

One commit = one logical change. Don't dump a whole week into a single commit.

---

## 4. What NOT to commit

The `.gitignore` already blocks these, but be aware:

- `__pycache__/`, `*.pyc`
- `logs/`, `wandb/`, `outputs/`, `runs/` — training artifacts (these live on wandb)
- `*.pt`, `*.pth`, `*.ckpt` — model weights (too big; share via wandb or a link)

**USD files** (`assets/*.usd*`) **do** get committed — they're the vessel models and
the benchmark needs them. Keep them reasonably small (< ~5 MB each).

---

## 5. Keeping your branch fresh

If `main` moves while you work, rebase onto it before opening/updating your PR:

```bash
git checkout main && git pull
git checkout arif/catamaran-patrol
git rebase main
# fix any conflicts, then:
git push --force-with-lease
```

---

## 6. PR acceptance checklist

I'll merge your PR when it has:

- [ ] Task registered in `__init__.py` with a new gym id
- [ ] `cfg` file with physics constants commented (mass, volume, damping, thrust)
- [ ] env file with each reward term commented
- [ ] `STARTER_TASK.md` in the task folder
- [ ] ≥ 3 wandb runs (seeds 42/123/456, 3000+ iter) with links in the PR description
- [ ] Reproducible: I can clone, run, get similar numbers
- [ ] Body-axis "forward" direction verified and noted in the cfg

---

## 7. When something breaks

- **git conflict you can't resolve** → don't force anything, message me with the output.
- **pushed something by mistake** → tell me, don't try to rewrite shared history yourself.
- **not sure if a change belongs in `main`** → put it in a branch + PR and ask.

When in doubt: branch, push, ask. Branches are cheap and nothing on a branch can hurt `main`.
