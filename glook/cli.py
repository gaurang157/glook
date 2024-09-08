import os
import subprocess
import requests
import pkg_resources
import click
import re
# from .GView import main2

def create_header(title, width=35):
    """Create a fancy header with centered title."""
    title = f" {title} "
    total_padding = width - len(title) - 2
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    header = f"╭{'─' * (width - 2)}╮"
    header += f"\n│{' ' * left_padding}{title}{' ' * (right_padding + 15)}│"
    header += f"\n╰{'─' * (width - 2)}╯"
    return header

def center_text(text, width):
    """Center text within a given width without borders."""
    lines = text.split('\n')
    centered_lines = []
    for line in lines:
        padding = width - len(line)  # No border symbols
        if padding > 0:
            left_padding = padding // 2
            right_padding = padding - left_padding
            centered_lines.append(f"{' ' * left_padding}{line}{' ' * right_padding}")
        else:
            centered_lines.append(line[:width])
    return "\n".join(centered_lines)


def check_for_update(package_name):
    # Get the current version of the installed package
    current_version = pkg_resources.get_distribution(package_name).version

    # Get the latest version available on PyPi
    response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
    data = response.json()

    latest_version = data['info']['version']
    releases = data['info']
    # Extract the description field
    description = releases['description']

    # Define the regex pattern to capture the changelog section
    pattern = r'## CHANGELOG\b(.*?)(?=\n##|\Z)'

    # Search for the pattern in the description field
    match = re.search(pattern, description, re.DOTALL)

    # Check if a match was found and print the result
    if match:
        changelog_section = match.group(0).strip()
    else:
        changelog_section = "Changelog section not found."

    # Fancy header with centered title
    style_ = click.style("     Welcome to ", fg='magenta', bold=True) + click.style("G-Look ML!", fg='blue', bold=True)
    header = create_header(style_, width=35)

    if current_version < latest_version:
        # Get the release notes for the latest version
        latest_release_notes = changelog_section
        # Visual border
        border_width = 35
        top_border = click.style(f"╭{'─' * (border_width - 2)}╮", fg='cyan')
        bottom_border = click.style(f"╰{'─' * (border_width - 2)}╯", fg='cyan')
        
        # ASCII Art and Text
        click.echo(header)
        click.echo(top_border)
        click.echo(click.style(center_text(f"🎉 G-Look Update Available! 🎉\n", border_width), fg='magenta', bold=True))
        click.echo(click.style(center_text(f"New Version: {latest_version}\n", border_width), fg='yellow', bold=True))
        click.echo(click.style(center_text(latest_release_notes, border_width), fg='cyan'))
        click.echo(click.style("\n To update, run the command below:", fg='green', bold=True))
        click.echo(click.style(center_text(f"pip install --upgrade {package_name}", border_width), fg='blue', bold=True))
        click.echo(click.style("\n  Thank you for being awesome! 😎", fg='green'))
        click.echo(bottom_border)
        click.echo(click.style(center_text(f"⌛ G-Look will launch in few sec's ⌛.", 50), fg='bright_yellow', bold=True))
        
    else:
        click.echo(header)
        click.echo(click.style(center_text(f"✅ You are using the latest version of {package_name} {current_version}.", 50), fg='blue', bold=True))
        click.echo(click.style(center_text(f"⌛ G-Look will launch in few sec's ⌛.", 50), fg='bright_yellow', bold=True))
       

def main2():
    try:
        check_for_update('glook')
    except Exception as e:
        print(e)
    # Get the directory of the current module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "GLook.py")
    subprocess.call(["streamlit", "run", script_path])

if __name__ == "__main__":
    main2()
