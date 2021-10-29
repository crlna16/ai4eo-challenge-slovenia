import collections
import difflib
import itertools
import os
import os.path
import re
import requests
from textwrap import dedent
from typing import List, Tuple

from IPython.display import display, Markdown, HTML
from dotenv.main import find_dotenv, DotEnv


ENTITY_METADATA_URL = {
    "dev": "https://api.dev.hub.eox.at/metadata",
    "prod": "https://api.hub.eox.at/metadata",
}
CONTRIBUTIONS_API_URL = {
    "dev": "https://contributions-api.dev.hub.eox.at",
    "prod": "https://contributions-api.hub.eox.at",
}
RELEASE_URL = "https://contributions-api.hub.eox.at/base-image-release-notes/{tag}"


def _get_requirement_name_mapping(dev: bool):
    response = requests.get(ENTITY_METADATA_URL["dev" if dev else "prod"])
    response.raise_for_status()
    return {data["entity"]["id"]: data["entity"]["name"] for data in response.json()}


def _get_notebook(notebook_id: str, dev: bool):
    response = requests.get(
        f"{CONTRIBUTIONS_API_URL['dev' if dev else 'prod']}/notebooks/{notebook_id}"
    )
    response.raise_for_status()
    return response.json()


def prepare(notebook_id: str, dev: bool = False):
    # Legacy. This will be removed in an upcoming release.

    # actual setup
    dot_env = DotEnv(find_dotenv())
    dot_env.set_as_environment_variables()

    # the rest displays information
    requirement_name_mapping = _get_requirement_name_mapping(dev=dev)
    notebook = _get_notebook(notebook_id, dev=dev)

    requirements = [
        requirement_name_mapping.get(req, req)
        for req in notebook.get("requirements", [])
    ]

    info = dedent(
        f"""
        ***Notebook Title***  
        {notebook['name']}
        
        ***Notebook Description***  
        {notebook['description']}
        
        """
    )

    if requirements:
        info += dedent(
            """
            ***Notebook Dependencies***  
            This notebook requires an active subscription to:
            """
        )
        info += "".join(f"* {req}\n" for req in requirements)

    info += dedent(
        """
        ---------
        
        *API credentials have automatically been injected for your active subscriptions.*
        
        The following environment variables are now available:
        """
    )
    info += "".join(f"* `{k}`\n" for k in dot_env.dict().keys())
    info += "\n-------------\n"

    display(Markdown(info))


def print_info(notebook_id: str, dev: bool = False):
    """Shows nice info for shared notebooks."""
    requirement_name_mapping = _get_requirement_name_mapping(dev=dev)
    notebook = _get_notebook(notebook_id, dev=dev)

    requirements = [
        requirement_name_mapping.get(req, req)
        for req in notebook.get("requirements", [])
    ]

    info = dedent(
        f"""
        ***Notebook Title***  
        {notebook['name']}
        
        ***Notebook Description***  
        {notebook['description']}
        
        """
    )

    if requirements:
        info += dedent(
            """
            ***Notebook Dependencies***  
            This notebook requires an active subscription to:
            """
        )
        info += "".join(f"* {req}\n" for req in requirements)

    display(Markdown(info))


def _format_env_list(variables):
    groups = itertools.groupby(
        sorted(variables),
        # group by first part, e.g. "SH" or "GEODB"
        key=lambda var: var.split("_", 1)[0],
    )
    formatted_groups = [", ".join(f"`{v}`" for v in group) for _, group in groups]
    return "".join(f"* {line}\n" for line in formatted_groups)


def setup_environment_variables():
    """Called in every notebook to inject credentials to environment"""
    dot_env = DotEnv(find_dotenv())
    dot_env.set_as_environment_variables()
    info = (
        "API credentials have automatically been injected for your active subscriptions.  \n"
        + "The following environment variables are now available:\n"
        + _format_env_list(dot_env.dict().keys())
        + "\n"
    )

    user_dot_env_path = "~/custom.env"
    user_dot_env = DotEnv(os.path.expanduser(user_dot_env_path))
    # NOTE: override is not True by default in dotenv
    user_dot_env.set_as_environment_variables()
    user_vars = user_dot_env.dict()
    if user_vars:
        info += (
            f"The following additional environment variables have been loaded from `{user_dot_env_path}`:\n"
            + _format_env_list(user_vars.keys())
        )

    display(Markdown(info))


# Example: -ipython                   7.13.0           py37hc8dfbb8_2    conda-forge
conda_list_output_regex = (
    r"^[-+](?P<name>\S+)\s+(?P<major_version>\d+)\.\S+\s+\S+\s+\S+$"
)
# TODO: use semver for anything more complicated


def _get_release_message(tag: str) -> str:
    response = requests.get(RELEASE_URL.format(tag=tag))
    response.raise_for_status()
    return response.text


def _calculate_diff_lines(old: str, new: str) -> List[str]:
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), n=0, lineterm="")
    # remove position markers ('@@ -25,2 +25,2 @@')
    diff = [
        entry for entry in diff if not (entry.startswith("@@") and entry.endswith("@@"))
    ]
    # remove "header" (---, +++)
    diff = diff[2:]
    return diff


def _group_major_minor_added_remove(
    diff_lines: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Takes diff lines and returns them grouped by meaning of change"""
    libraries = collections.defaultdict(list)

    for line in diff_lines:
        match = re.match(conda_list_output_regex, line)
        if match:
            libraries[match.group("name")].append(
                {"major_version": match.group("major_version"), "line": line}
            )

    major, minor, added, removed = [], [], [], []
    for lib, versions in libraries.items():

        if len(versions) == 2:
            if versions[0]["major_version"] != versions[1]["major_version"]:
                major.extend((versions[0]["line"], versions[1]["line"]))
            else:
                minor.extend((versions[0]["line"], versions[1]["line"]))

        elif len(versions) == 1:
            sign = versions[0]["line"][0]
            if sign == "-":
                removed.append(versions[0]["line"])
            elif sign == "+":
                added.append(versions[0]["line"])

    return major, minor, added, removed


def _append_nl(seq: List[str]) -> List[str]:
    return [f"{i}\n" for i in seq]


def check_compatibility(tag: str, dependencies: List[str] = None) -> None:
    _check_base_image_tag(tag)
    if dependencies:
        _check_dependencies(dependencies)


def _check_base_image_tag(tag: str) -> None:
    display(
        HTML(
            """<script type="text/javascript">
        function toggle(id) {
            el = document.getElementById(id);
            el.style.display = el.style.display === "none" ? "block" : "none";
        }
    </script>"""
        )
    )

    current_image_tag = "user-" + os.environ["JUPYTER_IMAGE"].split(":", 2)[1]
    if current_image_tag == tag:
        msg = f"This notebook is compatible with this base image version ({tag})."
    else:
        msg = dedent(
            f"""
            <h2>Base image difference detected</h2>
            <p>This notebook has been verified using the <a href="https://github.com/eurodatacube/base-images/releases/tag/{tag}" target="_blank">base image <strong>{tag}</strong></a>,
            whereas you are currently running <a href="https://github.com/eurodatacube/base-images/releases/tag/{current_image_tag}" target="_blank">base image <strong>{current_image_tag}</strong></a>.</p>

            <p>If you experience any problems, please consult the <a href="https://eurodatacube.com/marketplace" target="_blank">marketplace</a> for a more recent version of this notebook.</p>

            <p>The following changes occurred in base image in between these versions:</p>

            """
        )
        notebook_image_release_msg = _get_release_message(tag)
        current_image_release_msg = _get_release_message(current_image_tag)

        diff_lines = _calculate_diff_lines(
            old=notebook_image_release_msg,
            new=current_image_release_msg,
        )

        major, minor, added, removed = _group_major_minor_added_remove(diff_lines)

        def format_section(heading, entries, count):
            elm_id = id(heading)

            return f"""
            <p><h3 style="display: inline">{heading} ({count})</h3>
            <a href="#" onclick="toggle('#{elm_id}')">show</a><p/>
            <pre id="#{elm_id}" style="display: none; margin-top: 5px"><code>{''.join(_append_nl(entries))}</code></pre>
            """

        if added:
            msg += format_section("New libraries", added, len(added))

        if removed:
            msg += format_section("Removed libraries", removed, len(removed))

        if major:
            msg += format_section(
                "Libraries with major version differences", major, int(len(major) / 2)
            )

        if minor:
            msg += format_section(
                "Libraries with minor version differences", minor, int(len(minor) / 2)
            )

    display(HTML(msg))


def _check_dependencies(dependencies: List[str]) -> None:
    envs_by_dependency = {
        dependency: ", ".join(
            f"`{k}`" for k in os.environ if k.startswith(f"{dependency}_")
        )
        for dependency in dependencies
    }

    info = (
        dedent(
            """

    ---------

    The following environment variables are available:

    """
        )
        + "".join(f"* {envs}\n" for envs in envs_by_dependency.values() if envs)
    )

    for dep, env in envs_by_dependency.items():
        if not env:
            info += f"\n-------------\nNo environment variables for **{dep}** found.\n"

    display(Markdown(info))
