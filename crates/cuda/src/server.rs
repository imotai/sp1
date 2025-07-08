#![allow(unused)]

#[cfg(feature = "native")]
mod native {
    use crate::client::CudaClientError;
    use std::{path::PathBuf, process::Stdio};
    use tokio::{io::AsyncWriteExt, process::Command};

    /// Install a systemd unit for the given CUDA device id, and try to start it.
    ///
    /// Note: This method may cause race conditions, it should be called in a critical section.
    pub(crate) async fn start_server(cuda_id: u32) -> Result<(), CudaClientError> {
        assert_is_systemd().await;

        maybe_download_server(cuda_id).await?;
        maybe_install_systemd_unit(cuda_id).await?;

        let mut cmd = Command::new("systemctl");
        // Ensure the unit is known to the system.
        cmd.arg("--user").arg("daemon-reload").status().await.map_err(CudaClientError::Connect)?;

        // Enable the unit.
        let mut cmd = Command::new("systemctl");
        cmd.arg("--user")
            .arg("enable")
            .arg("--now")
            .arg(unit_name(cuda_id))
            .status()
            .await
            .map_err(CudaClientError::Connect)?;

        let output = cmd.output().await.map_err(CudaClientError::Connect)?;
        if !output.status.success() {
            return Err(CudaClientError::Connect(std::io::Error::other(String::from_utf8_lossy(
                &output.stderr,
            ))));
        }

        Ok(())
    }

    async fn maybe_install_systemd_unit(cuda_id: u32) -> Result<(), CudaClientError> {
        const SYSTEMD_TEMPLATE: &str = include_str!("cuslop.service.template");
        const USR_SERVICE_PATH: &str = ".config/systemd/user";

        let home = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
            std::env::var("HOME").expect("$HOME or $XDG_CONFIG_HOME is not set.")
        });

        let unit_path = PathBuf::from(home)
            .join(USR_SERVICE_PATH)
            .join(format!("{}.service", unit_name(cuda_id)));

        tracing::debug!("Installing cuslop-server systemd unit at {}", unit_path.display());

        if unit_path.exists() {
            return Ok(());
        }

        let unit_content = SYSTEMD_TEMPLATE.replace("{%}", &cuda_id.to_string());

        tokio::fs::create_dir_all(unit_path.parent().unwrap())
            .await
            .map_err(CudaClientError::Connect)?;

        let mut file =
            tokio::fs::File::create(unit_path).await.map_err(CudaClientError::Connect)?;

        file.write_all(unit_content.as_bytes()).await.map_err(CudaClientError::Connect)?;

        Ok(())
    }

    /// Asserts that the system is using systemd.
    async fn assert_is_systemd() {
        let mut cmd = Command::new("which");
        let status = cmd
            .stdout(Stdio::null())
            .arg("systemctl")
            .status()
            .await
            .expect("`which systemctl` command failed");

        if !status.success() {
            panic!("only systemd is supported for native mode");
        }
    }

    // If the server binary is not found in the path, or if it the version is not compatible,
    // download the server binary from the release page.
    async fn maybe_download_server(cuda_id: u32) -> Result<(), CudaClientError> {
        // Check if the server binary is in the path.
        let mut download = false;
        let home = std::env::var("HOME").expect("$HOME is not set");
        let path = PathBuf::from(home).join(".sp1/bin/cuslop-server");
        if !path.exists() {
            download = true;
        } else {
            let version = Command::new(&path)
                .arg("--version")
                .output()
                .await
                .map_err(CudaClientError::DownloadIO)?;

            let version = String::from_utf8_lossy(&version.stdout);

            if version != sp1_primitives::SP1_VERSION {
                download = true;

                // Stop *ALL* services, so we can replace it with a new version.
                //
                // NOTE: If a user is running a CUDA prover, across different versions,
                // on the same machine, this will cause other instances to crash!
                let mut cmd = Command::new("systemctl");
                cmd.arg("--user").arg("stop").arg("'cuslop-server-*'");

                let _ = cmd.status().await.map_err(CudaClientError::DownloadIO)?;
            }
        }

        if download {
            let version = format!("v{}", sp1_primitives::SP1_VERSION);

            // todo!(nathan): sp1-wip -> sp1
            // note this doesnt work since wip is not public we would need to add an auth token.
            let url = format!(
                "https://github.com/succinctlabs/sp1-wip/releases/download/{version}/cuslop_server_{version}_x86_64.tar.gz",
            );

            tracing::debug!("Downloading CUDA server from {}", url);

            // Download the tar file.
            let tar_file = path.with_extension("tar.gz");
            let mut file =
                tokio::fs::File::create(&tar_file).await.map_err(CudaClientError::DownloadIO)?;

            let client = reqwest::Client::new();

            let mut request = client.get(url);
            // We add the token to the request if it is set, this is used when were working in the
            // private repo.
            if let Ok(token) = std::env::var("GH_TOKEN") {
                request = request.bearer_auth(token);
            }
            let mut response = request.send().await.map_err(CudaClientError::Download)?;

            if !response.status().is_success() {
                tracing::error!("Bad status code from download attempt: {}", response.status());

                return Err(CudaClientError::Unexpected(format!(
                    "Failed to download CUDA server: {}",
                    response.text().await.expect("failed to read response text")
                )));
            }

            let bytes = response.bytes().await?;
            file.write_all(&bytes).await.map_err(CudaClientError::DownloadIO)?;

            // Extract the tar file.
            let mut cmd = Command::new("tar");
            cmd.arg("-xzf").arg(&tar_file).arg("-C").arg(path.parent().unwrap());
            cmd.status().await.map_err(CudaClientError::DownloadIO)?;

            // Remove the tar file.
            tokio::fs::remove_file(tar_file).await.map_err(CudaClientError::DownloadIO)?;
        }

        Ok(())
    }

    #[allow(clippy::uninlined_format_args)]
    /// The name of the systemd unit for the given CUDA device id.
    fn unit_name(cuda_id: u32) -> String {
        format!("cuslop-server-{}", cuda_id)
    }
}

#[cfg(all(feature = "native", target_arch = "x86_64"))]
pub(crate) use native::start_server;

#[cfg(any(not(feature = "native"), not(target_arch = "x86_64")))]
mod docker {
    use crate::client::CudaClientError;
    use std::process::Stdio;
    use tokio::process::Command;

    /// Start the docker server.
    ///
    /// Note this method *will fail* if ran twice with the same `cuda_id`.
    ///
    /// This method should only be called in a critical section
    pub(crate) async fn start_server(cuda_id: u32) -> Result<(), CudaClientError> {
        let image =
            format!("public.ecr.aws/succinct-labs/cuslop-server:v{}", sp1_primitives::SP1_VERSION);

        if let Err(e) = Command::new("docker").args(["pull", &image]).output().await {
            return Err(CudaClientError::Unexpected(format!(
                "Failed to pull Docker image: {e}. Ensure docker is installed and running."
            )));
        }

        // Just log any errors the result, if the container is already running, this will fail.
        //
        // If the container failed to start for whatver reason, the logs are piped to stdio,
        // and we will see the error, we will explicitly throw during the connection phase next.
        match Command::new("docker")
            .args([
                "run",
                "-e",
                &format!("RUST_LOG={}", "debug"),
                "-e",
                "CUDA_VISIBLE_DEVICES",
                &cuda_id.to_string(),
                // Remove the container on exit.
                "--rm",
                // Share the tmp directory with the container.
                // This is where the socket will be created.
                "-v",
                "/tmp:/tmp",
                // Use all GPUs.
                "--gpus",
                "all",
                // The name of the container.
                "--name",
                format!("cuslop-server-{cuda_id}").as_str(),
                // The image to run.
                &image,
            ])
            // Redirect stdout and stderr to the parent process
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .await
        {
            Ok(status) => {
                if !status.success() {
                    // Its possible the container is already running, so we ignore the error.
                    tracing::debug!(
                        "Failed to start new Docker container for CUDA device {}: {}",
                        cuda_id,
                        status
                    );
                }
            }
            Err(e) => {
                return Err(CudaClientError::Unexpected(format!(
                    "Failed to start new Docker container for CUDA device {cuda_id}: {e}"
                )));
            }
        }

        Ok(())
    }
}

#[cfg(any(not(feature = "native"), not(target_arch = "x86_64")))]
pub(crate) use docker::start_server;
